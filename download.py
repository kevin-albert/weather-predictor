""" Download radar data from public S3 bucket """

from os.path import join, split, exists
from dateutil.parser import parse

import numpy as np
import matplotlib.pyplot as plt
import pyart
import boto3

S3 = boto3.resource('s3')

def fetch_data_for_date(record_date, site_id, data_dir):
    """
    Download radar data for a given date and siteId from public S3 bucket
    """
    files = []
    bucket = S3.Bucket('noaa-nexrad-level2')
    data_site_prefix = '{record_date}/{site_id}'.format(record_date=record_date, site_id=site_id)
    for s3_object in bucket.objects.filter(Prefix=data_site_prefix):
        filename = split(s3_object.key)[1]
        filepath = join(data_dir, filename)
        if not exists(filepath):
            print('Downloading {filename} to {filepath}'.format(
                filename=filename, filepath=filepath))
            bucket.download_file(s3_object.key, filepath)
        files.append(filepath)

    return files

def radar_from_file(filepath):
    """
    Load a radar object from a file
    """

    return pyart.io.read_nexrad_archive(filepath)

def grid_from_radar_field(radar, field='reflectivity'):
    """
    Create a Cartesian grid from some radar data
    """

    # mask out last 10 gates of each ray, this removes the "ring" around the radar.
    radar.fields[field]['data'][:, -10:] = np.ma.masked

    # exclude masked gates from the gridding
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_transition()
    gatefilter.exclude_masked(field)

    # perform Cartesian mapping, limit to the reflectivity field.
    grid = pyart.map.grid_from_radars(
        (radar,), gatefilters=(gatefilter, ),
        grid_shape=(1, 241, 241),
        grid_limits=((2000, 2000), (-123000.0, 123000.0), (-123000.0, 123000.0)),
        fields=[field])

    return grid

def download_and_save(record_date, site_id, data_dir):
    """
    Start the download
    """

    files = fetch_data_for_date(record_date, site_id, 'level2data')

    field = 'reflectivity'
    for radar_file in files:
        print('Reading {0}'.format(radar_file))
        radar = radar_from_file(radar_file)
        print('Creating grid from radar data')
        radar_grid = grid_from_radar_field(radar, field)
        radar_time = pyart.graph.common.generate_grid_time_begin(radar_grid)

        filepath = join(data_dir, '{0}-{1}.nc4'.format(site_id, radar_time))
        print('Writing to {0}'.format(filepath))
        pyart.io.write_grid(filepath, radar_grid)


def load_test():
    """ Testing """

    field = 'reflectivity'
    filepath = 'data/KMUX-2016-12-15 00:00:14.615000.nc4'
    radar_grid = pyart.io.read_grid(filepath)

    # create the plot
    fig = plt.figure()
    radar_plt = fig.add_subplot(111)
    radar_plt.imshow(radar_grid.fields[field]['data'][0], origin='lower')
    plt.show()


download_and_save('2016/12/16', 'KMUX', 'data')
# load_test()

# # create the plot
# fig = plt.figure()
# radar_plt = fig.add_subplot(111)
# radar_plt.imshow(radar_grid.fields[field]['data'][0], origin='lower')
# plt.show()