from landsat import search
from landsat.image import Simple, PanSharpen, FileDoesNotExist
from landsat.ndvi import NDVIWithManualColorMap, NDVI
import pickle
import urllib2
import sys
import argparse
import json
import os
import datetime

FIRE_LABEL_FILE='data/landsat_fire_2016.csv'
BASE = "/Volumes/passport/"
DEFAULT_DOWNLOAD_PATH=os.path.join(BASE,'landsat_download_2016')
DEFAULT_PROCESSED_PATH=os.path.join(BASE,'landsat_processed_2016')
REMOVE_AFTER_PROCESS=False
DOWNLOAD_BANDS=[1,2,3,4,5,6,7]
DOWNLOAD_ADDITIONAL_FILES=["_BQA.TIF", "_MTL.txt"]
DOWNLOAD_BASE_URL='https://landsat-pds.s3.amazonaws.com/c1/L8'
DOWNLOAD_PRE_COLLECTION_BASE_URL='https://landsat-pds.s3.amazonaws.com/L8'
CLOUD_FILTER=30
GEO_FENCE = [-66.951381, -124.7844079, 24.7433195, 49.3457868] #lon, lat
DOWNLOAD_ONLY = True

parser = argparse.ArgumentParser()
parser.add_argument('--target_csv', default=FIRE_LABEL_FILE, help="Fire label file")
parser.add_argument('--download_dir', default=DEFAULT_DOWNLOAD_PATH, help="Directory with the SIGNS dataset")
#parser.add_argument('--processed_dir', default=DEFAULT_PROCESSED_PATH, help="Where to write the new data")

bad_loc_cache = []
good_loc_cache = {}
def make_key(lon, lat, start_date):
    # round up cache to decimal
    return start_date + str(round(lon)) + str(round(lat))

def get_image_ID(lon, lat, start_date, end_date=None, cloud_threshold=99):
    if make_key(lon, lat, start_date) in bad_loc_cache:
        #print "foudn bad loc in cache"
        return -1, -1, -1, None
    if make_key(lon, lat, start_date) in good_loc_cache:
        #print "found good loc in cache"
        return good_loc_cache[make_key(lon, lat, start_date)]
    s = search.Search()
    #print "search image for lon {}, lat{}, date {}".format(lon, lat, start_date)
    try:
        result = s.search(lon=lon, lat=lat, start_date=start_date, end_date=end_date)
    except:
        print "failed to retrieve serach result, skip!"
        result = {}
        return -1, -1, -1, None
    if result != {} and result['status'] != 'SUCCESS':
        return -1, -1, -1, None

    scene_id = -1
    download_key = -1
    cloud = -1
    min_cloud = 100
    fire_date = None
    try:
        for r in result["results"]:
            # chose the lowest cloud coverage
            if r['cloud'] < cloud_threshold and r['cloud'] < min_cloud and not r['thumbnail'].split("/")[8].split(".")[0].endswith("T2"):
                scene_id = r['sceneID']
                # extract download key from thumbnail url
                download_key = r['thumbnail'].split("/")[8].split(".")[0]
                cloud = r['cloud']
                min_cloud = cloud
                fire_date = r['date']
            else:
                bad_loc_cache.append(make_key(lon, lat, start_date))
    except:
        print result
        print "==== DEBUG : weird key error!!!! ===="

    good_loc_cache[make_key(lon, lat, start_date)] = (scene_id, download_key, cloud, fire_date)
    return scene_id, download_key, cloud, fire_date

def preprocess(download_key, src_path=DEFAULT_DOWNLOAD_PATH, dst_path=DEFAULT_PROCESSED_PATH,
               ndvi=False, pansharpen=False, verbose=False, ndvigrey=False, bounds=None):

    try:
        bands = [2,3,4]
        if pansharpen:
            p = PanSharpen(src_path, bands=bands, dst_path=dst_path,
                           verbose=verbose,  bounds=bounds)
        elif ndvigrey:
            p = NDVI(src_path, verbose=verbose, dst_path=dst_path, bounds=bounds)
        elif ndvi:
            p = NDVIWithManualColorMap(src_path, dst_path=dst_path,
                                       verbose=verbose, bounds=bounds)
        else:
            p = Simple(src_path, bands=bands, dst_path=dst_path, verbose=verbose, bounds=bounds)

    except IOError as err:
        print str(err)
        exit(str(err))
    except FileDoesNotExist as err:
        print str(err)
        exit(str(err))

    return p.run()



def download_scene(download_key, savepath_base=DEFAULT_DOWNLOAD_PATH, aws=False, pre_collection = False):
    if not os.path.exists(savepath_base):
        print "Download path {} does not exit".format(savepath_base)
        exit(1)
    # start downloading
    savepath = os.path.join(savepath_base, download_key)
    if os.path.exists(savepath):
        print "Download directory for {} already exists!, skip download".format(savepath)
        return
    else:
        os.mkdir(savepath)
    print "[Downloading Image] : {}".format(download_key)

    #collection_number = sceene_id[]
    if pre_collection:
        download_key = download_key[:-1] + "0"
        WRS_path = download_key[3:6]
        WRS_row = download_key[6:9]
        #https://s3-us-west-2.amazonaws.com/landsat-pds/L8/034/026/LC80340262016269LGN00/index.html

        url_base = DOWNLOAD_PRE_COLLECTION_BASE_URL + "/{}/{}/{}/".format(WRS_path, WRS_row, download_key)
    else:
        WRS_path = download_key.split("_")[2][:3]
        WRS_row = download_key.split("_")[2][3:]
        url_base = DOWNLOAD_BASE_URL + "/{}/{}/{}/".format(WRS_path, WRS_row, download_key)
    band_files = ["_B{}.TIF".format(b) for b in DOWNLOAD_BANDS]
    additional_files = DOWNLOAD_ADDITIONAL_FILES
    for file_ext in band_files + additional_files:
        url = url_base + download_key + file_ext
        print "Downloadign from {}".format(url)
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

def in_geo_fence(lon, lat):
    return lon < GEO_FENCE[0] and lon > GEO_FENCE[1] and lat > GEO_FENCE[2] and lat < GEO_FENCE[3]


def process_csv(csv_file, log_dir):
    with open(csv_file, 'r') as f:
        for i in range(1):
            f.readline()
        # skip first line
        print "start processing"
        meta_fire = {}
        scene_list = []
        tmp_meta_fire = {}
        tmp_scene_list = []
        count = 0
        image_count = 0
        line_count = 0
        for line in f:
            line_count += 1
            (fid, area, _1, _2, fire_id, lat, lon, raw_date, julian, gmt) = line.split(",")[:-5]
            # round up lon/lat to 0.01, ignore all duplicates at this level
            lon = round(float(lon), 2)
            lat = round(float(lat), 2)
            if not in_geo_fence(lon, lat):
                continue
            try:
                date = raw_date.split()[0].split("/")
                fire_date = datetime.datetime(int(date[2]), int(date[0]), int(date[1]))
                start_date = fire_date - datetime.timedelta(days=1)
                end_date = start_date + datetime.timedelta(days=1) * 3
                fire_date = "{0:0>2}-{1:0>2}-{2:0>2}".format(fire_date.year, fire_date.month, fire_date.day)
                start_date = "{0:0>2}-{1:0>2}-{2:0>2}".format(start_date.year, start_date.month, start_date.day)
                end_date = "{0:0>2}-{1:0>2}-{2:0>2}".format(end_date.year, end_date.month, end_date.day)
            except:
                print "Got empty/weird date from csv {}".format(raw_date)
                continue
            scene_id, download_key, cloud, image_date = get_image_ID(lon, lat, start_date=start_date,
                                                         end_date=end_date, cloud_threshold=CLOUD_FILTER)
            count += 1
            if scene_id != -1:
                if download_key not in meta_fire:
                    meta_fire[download_key] = {"lats_lons":[(lat, lon)], "image_date": image_date, "date":[fire_date],
                                               "scene_id":scene_id, "cloud":cloud, "fires":[fire_id]}
                else:
                    if (lat, lon) in meta_fire[download_key]["lats_lons"]:
                        continue
                    else:
                        meta_fire[download_key]["fires"].append(fire_id)
                        meta_fire[download_key]["lats_lons"].append((lat,lon))
                        meta_fire[download_key]["date"].append(start_date)
                if download_key not in scene_list:
                    scene_list.append(download_key)
                    print download_key
                    image_count += 1
            else:
                pass
                # keep downloading if this happen, prob cuz there are to much cloud
            if image_count >= 1000:
                break
            if count % 100 == 0:
                print "Status: serached {} fire, found {} images".format(count, image_count)
                # flush out data when running
                with open(os.path.join(log_dir, meta_name_base + "_meta.json"), 'w') as json_f:
                    json.dump(meta_fire, json_f, indent=4)
                with open(os.path.join(log_dir, meta_name_base + "_scene_lists.json"), 'w') as json_f:
                    json.dump(scene_list, json_f, indent=4)

        meta_fire["total_line_read"] = line_count
        with open(os.path.join(log_dir,  meta_name_base +"_meta.pickle"), 'w') as pickle_f:
            pickle.dump(meta_fire, pickle_f)
        with open(os.path.join(log_dir,  meta_name_base + "_scene_lists.pickle"), 'w') as pickle_f:
            pickle.dump(scene_list, pickle_f)
        with open(os.path.join(log_dir, meta_name_base+"_meta.json"), 'w') as json_f:
            json.dump(meta_fire, json_f, indent=4)
        with open(os.path.join(log_dir, meta_name_base+"_scene_lists.json"), 'w') as json_f:
            json.dump(scene_list, json_f, indent=4)
        print "====== Processed {} entries =====".format(line_count)
    return meta_fire, scene_list


if __name__ == '__main__':

    args = parser.parse_args()
    assert os.path.isdir(args.download_dir), "Couldn't find the dataset at {}".format(args.download_dir)
    meta_name_base = os.path.basename(args.target_csv).strip(".csv")

    if not DOWNLOAD_ONLY:
        meta_fire, scene_list = process_csv(args.target_csv, args.download_dir)
        #exit(1)
    else:
        with open(os.path.join(args.download_dir, meta_name_base+"_scene_lists.json"), 'rb') as json_f:
            scene_list = json.load(json_f)
        with open(os.path.join(args.download_dir, meta_name_base+"_meta.json"), 'rb') as json_f:
            meta_fire = json.load(json_f)

    print "====== Start downloading all {} images =====".format(len(scene_list))
    count = 0
    for download_key in scene_list:
        print "donwloading image {}, {}".format(count, download_key)
        try:
            collection_date = datetime.datetime.strptime(meta_fire[download_key]["image_date"], '%Y-%m-%d')
            pre_collection_threshold = datetime.datetime.strptime("2017-5-1", '%Y-%m-%d')
        except:
            print "Got weird datetime error : ", sys.exc_info()[0]
            scene_list.remove(download_key)
            del meta_fire[download_key]
            continue
        try:
            if collection_date < pre_collection_threshold:
                download_scene(meta_fire[download_key]["scene_id"], savepath_base=args.download_dir, pre_collection=True)
            else:
                download_scene(download_key, savepath_base=args.download_dir)
        except:
            print "Unexpected error:", sys.exc_info()[0]
            print "Failed to download image {}".format(download_key)
            scene_list.remove(download_key)
            del meta_fire[download_key]
        count+=1
    # rewrite meta files in case images fail to download
    with open(os.path.join(args.download_dir,  meta_name_base +"_meta.pickle"), 'w') as pickle_f:
        pickle.dump(meta_fire, pickle_f)
    with open(os.path.join(args.download_dir,  meta_name_base + "_scene_lists.pickle"), 'w') as pickle_f:
        pickle.dump(scene_list, pickle_f)
    with open(os.path.join(args.download_dir, meta_name_base+"_meta.json"), 'w') as json_f:
        json.dump(meta_fire, json_f, indent=4)
    with open(os.path.join(args.download_dir, meta_name_base+"_scene_lists.json"), 'w') as json_f:
        json.dump(scene_list, json_f, indent=4)
    exit(1)
    print "====== Start preprocessing all {} images =====".format(len(scene_list))
    for download_key in download_list:
        try:
            preprocess(download_key, src_path=os.path.join(args.download_dir, download_key),
                       dst_path=os.path.join(args.download_dir, download_key))
        except:
            print "====== cannot find image {} =====".format(download_key)





