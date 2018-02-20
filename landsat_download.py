from landsat import search
from landsat.image import Simple, PanSharpen, FileDoesNotExist
from landsat.ndvi import NDVIWithManualColorMap, NDVI
import pickle
import landsat
import os
import urllib2

FIRE_LABEL_FILE='landsat_fire_last7.csv'
DEFAULT_DOWNLOAD_PATH='landsat_download'
DEFAULT_PROCESSED_PATH='landsat_processed'
DOWNLOAD_BANDS=[2,3,4]
DOWNLOAD_ADDITIONAL_FILES=["_BQA.TIF", "_MTL.txt", "_ANG.txt"]
DOWNLOAD_BASE_URL='https://landsat-pds.s3.amazonaws.com/c1/L8'

#https://landsat-pds.s3.amazonaws.com/c1/L8/139/045/LC08_L1TP_139045_20170304_20170316_01_T1/index.html

def get_image_ID(lon, lat, start_date, end_date=None, cloud_threshold=100):
	s = search.Search()
	result = s.search(lon=lon, lat=lat, start_date=start_date)
	if result['status'] != 'SUCCESS':
		return -1, -1
	min_cloud = 100
	scene_id = -1
	download_key = -1
	cloud = -1
	for r in result["results"]:
		# chose the lowest cloud coverage
		if r['cloud'] < cloud_threshold and r['cloud'] < min_cloud:
			scene_id = r['sceneID']
			# extract download key from thumbnail url
			download_key = r['thumbnail'].split("/")[8].split(".")[0]
			cloud = r['cloud']
	return scene_id, download_key, cloud

def preprocess(download_key, base_downalod_path=DEFAULT_DOWNLOAD_PATH, savepath=DEFAULT_PROCESSED_PATH, ndvi=False, pansharpen=False, verbose=False, ndvigrey=False, bounds=None):
    path = os.path.join(base_download_path, download_key)
    try:
        bands = [2,3,4]
        if pansharpen:
            p = PanSharpen(path, bands=bands, dst_path=dst_path,
                           verbose=verbose, force_unzip=force_unzip, bounds=bounds)
        elif ndvigrey:
            p = NDVI(path, verbose=verbose, dst_path=dst_path, bounds=bounds)
        elif ndvi:
            p = NDVIWithManualColorMap(path, dst_path=dst_path,
                                       verbose=verbose, bounds=bounds)
        else:
            p = Simple(path, bands=bands, dst_path=dst_path, verbose=verbose, bounds=bounds)

    except IOError as err:
        exit(str(err), 1)
    except FileDoesNotExist as err:
        exit(str(err), 1)

    return p.run()


def download_scene(download_key, savepath_base=DEFAULT_DOWNLOAD_PATH, aws=False):
	if not os.path.exists(savepath_base):
		print "Download path {} does not exit".format(savepath_base)
		exit(1)
	# start downloading
	savepath = os.path.join(savepath_base, download_key)
	if os.path.exists(savepath):
		print "Download directory for {} already exists!"
	else:
		os.mkdir(savepath)
	print "[Download Image] : {}".format(download_key)

	WRS_path = download_key.split("_")[2][:3]
	WRS_row = download_key.split("_")[2][3:]
	#collection_number = sceene_id[]
	url_base = DOWNLOAD_BASE_URL + "/{}/{}/{}/".format(WRS_path, WRS_row, download_key)
	band_files = ["_B{}.TIF".format(b) for b in DOWNLOAD_BANDS]
	additional_files = DOWNLOAD_ADDITIONAL_FILES
	for file_ext in band_files + additional_files:
		url = url_base + download_key + file_ext
		print url
		u = urllib2.urlopen(url)
		file_name = os.path.join(savepath, url.split("/")[-1])
		f = open(file_name, 'wb')
		meta = u.info()
		file_size = int(meta.getheaders("Content-Length")[0])
		print "Downloading: %s Bytes: %s" % (file_name, file_size)
		file_size_dl = 0
		block_sz = 8192
		while True:
		    buffer = u.read(block_sz)
		    if not buffer:
		        break

		    file_size_dl += len(buffer)
		    f.write(buffer)
		    status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
		    status = status + chr(8)*(len(status)+1)
		    print status,
		f.close()
	# download other files

with open(FIRE_LABEL_FILE, 'r') as f:
	f.readline()
	# skip first line
	meta_fire = {}
	scene_list = []
	count = 0
	for line in f:
		count += 1
		(fid, area, _1, _2, fire_id, lat, lon, datetime, julian, gmt) = line.split(",")[:-5]
		lon = float(lon)
		lat = float(lat)
		date = datetime.split()[0].split("/")
		#date = "2017-{0:0>2}-{1:0>2}".format(date[0], date[1])
		date = "20{0:0>2}-{1:0>2}-{2:0>2}".format(date[2], date[0], date[1])
		scene_id, download_key, cloud = get_image_ID(float(lon), float(lat), date)
		if scene_id != -1:
			meta_fire[fire_id] = (lon, lat, date, scene_id, download_key, cloud)
			if scene_id not in scene_list:
				scene_list.append((download_key))
		else:
			print "fail to retrieve fire {} with {}".format(fire_id, (lon, lat, date)) 
	pickle.dump(meta_fire, FIRE_LABEL_FILE.strip(".csv") +"_meta.pickle")
	pickle.dump(scene_list, landsat_fire_last7+"_scene_list.pickle")
	# now download all scene image
	for image in scene_list: 
		download_scene(image)

	for image in scene_list:
		preprocess(image)





