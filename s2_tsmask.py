import shutil
from collections import Counter
from collections.abc import Sequence, Mapping
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import logging
import uuid

import dask
from dask.distributed import Client
import click
import numpy
import xarray
import mock
import rasterio
from affine import Affine
import yaml
from yaml import SafeDumper
import zarr

import datacube
from datacube.model import DatasetType as Product
from datacube.model.utils import make_dataset
from datacube.virtual import catalog_from_file, Measurement, Transformation
from datacube.virtual import DEFAULT_RESOLVER
from datacube.virtual.impl import VirtualDatasetBag
from datacube.utils.geometry import GeoBox, CRS, unary_union

from dea_tsmask import tsmask_temporal, spatial_noise_filter


MINIMUM_REQUIRED_OBSERVATIONS = 90
PRODUCT_NAME = 's2_tsmask'


def init_logging():
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(asctime)s - %(message)s')


def bag_filter(dataset_bag, predicate):
    def worker(bag):
        if isinstance(bag, Sequence):
            return [ds for ds in bag if predicate(ds)]

        if isinstance(bag, Mapping):
            return {key: [worker(branch) for branch in value]
                    for key, value in bag.items()}

        raise ValueError('malformed bag')

    return VirtualDatasetBag(worker(dataset_bag.bag), dataset_bag.geopolygon, dataset_bag.product_definitions)


def custom_native_geobox(ds, measurements=None):
    metadata = ds.metadata_doc['image']['bands']['nbart_swir_3']['info']
    geotransform = metadata['geotransform']
    crs = CRS(ds.metadata_doc['grid_spatial']['projection']['spatial_reference'])
    affine = Affine(geotransform[1], 0.0, geotransform[0], 0.0, geotransform[5], geotransform[3])
    return GeoBox(width=metadata['width'], height=metadata['height'], affine=affine, crs=crs)


class TSMask(Transformation):
    def compute(self, data):
        chunks = dict(time=-1)
        result = xarray.apply_ufunc(tsmask_temporal,
                                    data.avg.chunk(chunks), data.mndwi.chunk(chunks), data.msavi.chunk(chunks),
                                    dask='parallelized', output_dtypes=['uint8']).assign_attrs(nodata=0, units='1')
        return result.to_dataset(name='classification').assign_attrs(**data.attrs)

    def measurements(self, input_measurements):
        return {'classification': Measurement(name='classification', dtype='uint8', nodata=0, units='1')}


def same_grid(input_bag):

    def tuplify(geobox):
        return ((geobox.width, geobox.height), tuple(geobox.affine))

    datasets = list(input_bag.contained_datasets())
    if not datasets:
        raise ValueError('no data')

    counter = Counter([tuplify(custom_native_geobox(ds)) for ds in datasets])
    common = counter.most_common(1)[0][0]

    def predicate(ds):
        return tuplify(custom_native_geobox(ds)) == common

    return bag_filter(input_bag, predicate)


def search(dc, region_code, product, mode):
    if mode == 'test':
        query = dict(region_code=region_code, time=('2018-06', '2018-07'))
    else:
        query = dict(region_code=region_code)

    try:
        bag = same_grid(product.query(dc, **query))
    except ValueError:
        raise ValueError('no data for {}'.format(region_code))

    if mode == 'test':
        return bag

    if len(list(bag.contained_datasets())) < MINIMUM_REQUIRED_OBSERVATIONS:
        raise ValueError(f'not enough datasets for {region_code}')

    if mode == 'initial':
        return bag

    now = datetime.utcnow().astimezone(timezone.utc)
    start = now.replace(year=now.year - 1)
    filtered = bag_filter(bag, lambda ds: ds.key_time > start)

    while len(list(filtered.contained_datasets())) < MINIMUM_REQUIRED_OBSERVATIONS:
        start = start - timedelta(days=6 * 30)
        filtered = bag_filter(bag, lambda ds: ds.key_time > start)

    return filtered


def create_datacube_product(dc, product_definition, measurements):
    metadata_type = dc.index.metadata_types.get_by_name(product_definition['metadata_type'])

    assert 'measurements' not in product_definition, 'unexpected measurement specs in product definition'

    result = {'measurements': list(measurements.values()), **product_definition}

    Product.validate(result)
    return Product(metadata_type, result)


def load_catalog():
    resolver = DEFAULT_RESOLVER.clone()
    resolver.register('transform', 'tsmask', TSMask)

    return catalog_from_file(str(Path(__file__).parent / 'virtual-product-catalog.yaml'),
                             name_resolver=resolver)


def write_timeslice(zarr_file, index, dtype, nodata, geobox, about, out_file):
    profile = {
        'compress': 'deflate',
        'driver': 'GTiff',
        'interleave': 'band',
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        'dtype': dtype,
        'nodata': nodata,
        'width': geobox.width,
        'height': geobox.height,
        'transform': geobox.affine,
        'crs': geobox.crs.crs_str,
        'count': 1
    }

    tmp_out_file = str(out_file) + '-tmp-' + str(uuid.uuid1())
    with rasterio.open(tmp_out_file, mode='w', **profile) as out:
        src = zarr.open(zarr_file)
        mask = src['classification'][index, ...]
        less_noisy_mask = spatial_noise_filter(spatial_noise_filter(numpy.expand_dims(mask, axis=0)))
        out.write(less_noisy_mask[0], 1)
        out.update_tags(**about)

    shutil.move(tmp_out_file, out_file)
    logging.info('wrote %s', out_file)


@click.command()
@click.option('--region-code', type=str, required=True)
@click.option('--mode', type=click.Choice(['initial', 'regular', 'test']), required=True)
@click.option('--outdir', type=click.Path(dir_okay=True, file_okay=False), required=True)
@click.option('--workers', type=int)
@click.option('--tmpdir', type=click.Path(dir_okay=True, file_okay=False))
@click.option('--cleanup', type=bool, default=True)
def main(region_code, mode, outdir, workers, tmpdir, cleanup):
    if workers is None:
        if mode == 'initial':
            workers = 4
            dask_chunks = dict(time=-1, x=400, y=400)
            memory_limit = '50GB'
        else:
            workers = 16
            dask_chunks = dict(time=-1, x=500, y=500)
            memory_limit = '16GB'

    if tmpdir is None:
        if 'PBS_JOBFS' in os.environ:
            tmpdir = os.environ['PBS_JOBFS']
        else:
            tmpdir = os.getcwd()

    generate_s2_tsmask(region_code, mode, outdir, workers, tmpdir, dask_chunks, memory_limit, cleanup)


def valid_region(dataset_box, index):
    item = dataset_box.box[index].item()
    datasets = item['collate'][1]
    return unary_union(ds.extent for ds in datasets)


def done(out_file):
    return out_file.exists() and out_file.with_suffix('.yaml').exists()


def output_file(folder, epoch):
    return folder / ('s2_tsmask_' + str(epoch).replace(':', '').replace('-', '')[:15] + '.tif')


def generate_s2_tsmask(region_code, mode, outdir, workers, tmpdir, dask_chunks, memory_limit, cleanup):
    init_logging()

    catalog = load_catalog()

    product = catalog[PRODUCT_NAME]
    description = catalog.describe(PRODUCT_NAME)

    dc = datacube.Datacube()

    dataset_bag = search(dc, region_code, product, mode)

    measurements = product.output_measurements(dataset_bag.product_definitions)
    lineage = list(dataset_bag.contained_datasets())

    datacube_product = create_datacube_product(dc,
                                               description['product_definition'],
                                               measurements)

    logging.info('processing %r', dataset_bag)

    box = product.group(dataset_bag)

    zarr_file = str(Path(tmpdir) / 'output_tsmask.zarr')
    if Path(zarr_file).exists():
        logging.info('found existing zarr file, removing')
        shutil.rmtree(zarr_file)

    with mock.patch('datacube.virtual.impl.native_geobox', side_effect=custom_native_geobox):
        data = product.fetch(box, dask_chunks=dask_chunks)

    # TODO netcdfy data
    # zarr does not like these
    attrs = dict(**data.attrs)
    time_attrs = dict(**data.coords['time'].attrs)
    data.coords['time'].attrs = {}
    data.attrs = {}

    folder = Path(outdir) / region_code
    folder.mkdir(exist_ok=True, parents=True)

    if all(done(output_file(folder, epoch)) for index, epoch in enumerate(data.coords['time'].values)):
        logging.info('all targets already completed!')
        return

    with dask.config.set({'distributed.admin.log-format': '%(name)s - %(levelname)s - %(asctime)s - %(message)s',
                          'distributed.client.heartbeat': '20s',
                          'distributed.comm.timeouts.connect': '60s',
                          'distributed.comm.timeouts.tcp': '300s',
                          'distributed.worker.memory.terminate': False,
                          'logging': {'distributed': 'info', 'distributed.client': 'info', 'distributed.worker': 'info'}}):

        logging.info(yaml.dump(dask.config.config, indent=4))

        with Client(n_workers=workers, processes=True, local_directory=tmpdir, memory_limit=memory_limit, threads_per_worker=1) as client:
            client.run(init_logging)

            client.scatter(data, broadcast=True)

            data.to_zarr(zarr_file)

    logging.info('finished processing tsmask')
    data.attrs = attrs
    data.coords['time'].attrs = time_attrs

    futures = []
    # this is cheap, let the local cluster figure it out
    with Client(local_directory=tmpdir) as client:
        for index, epoch in enumerate(data.coords['time'].values):
            out_file = output_file(folder, epoch)
            if done(out_file):
                logging.info("not writing %s: it's already there", out_file)
            else:
                dataset = make_dataset(product=datacube_product,
                                       sources=lineage,
                                       extent=data.geobox.extent,
                                       valid_data=valid_region(box, index),
                                       center_time=epoch.item(),
                                       band_uris={'classification': {'layer': 1,
                                                                     'path': out_file.as_uri()}})
                dataset.metadata_doc['provider'] = {'reference_code': region_code}
                with open(out_file.with_suffix('.yaml'), 'w') as fl:
                    fl.write(yaml.dump(dataset.metadata_doc, Dumper=SafeDumper))

                futures.append(client.submit(write_timeslice, zarr_file, index,
                                             data['classification'].dtype,
                                             data['classification'].nodata,
                                             data.geobox,
                                             description['about'],
                                             out_file))
        client.gather(futures)

    logging.info('finished packaging')
    if Path(zarr_file).exists() and cleanup:
        shutil.rmtree(zarr_file)

    if all(done(output_file(folder, epoch)) for index, epoch in enumerate(data.coords['time'].values)):
        with open(str(Path(outdir) / (region_code + '.done')), 'w') as fl:
            pass


if __name__ == '__main__':
    main()
