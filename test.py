import numpy as np
import matplotlib.pyplot as plt
import pyart
import boto3
import tempfile

# Let's use Amazon S3
s3 = boto3.resource('s3')
bucket = s3.Bucket('noaa-nexrad-level2')
s3key = '2016/12/15/KMUX/KMUX20161215_104149_V06'

# download to a local file-like object, and read it
localfile = tempfile.SpooledTemporaryFile()
bucket.download_fileobj(s3key, localfile)
localfile.seek(0)
radar = pyart.io.read_nexrad_archive(localfile)
localfile.close()

# mask out last 10 gates of each ray, this removes the "ring" around the radar.
radar.fields['reflectivity']['data'][:, -10:] = np.ma.masked

# exclude masked gates from the gridding
gatefilter = pyart.filters.GateFilter(radar)
gatefilter.exclude_transition()
gatefilter.exclude_masked('reflectivity')

# perform Cartesian mapping, limit to the reflectivity field.
grid = pyart.map.grid_from_radars(
    (radar,), gatefilters=(gatefilter, ),
    grid_shape=(1, 241, 241),
    grid_limits=((2000, 2000), (-123000.0, 123000.0), (-123000.0, 123000.0)),
    fields=['reflectivity'])

# create the plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(grid.fields['reflectivity']['data'][0], origin='lower')
plt.show()