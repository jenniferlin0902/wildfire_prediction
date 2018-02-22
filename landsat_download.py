from landsat import search
from landsat.image import Simple, PanSharpen, FileDoesNotExist
from landsat.ndvi import NDVIWithManualColorMap, NDVI
import pickle
import urllib2

import argparse
import json
import os

FIRE_LABEL_FILE='data/landsat_fire_last7.csv'
DEFAULT_DOWNLOAD_PATH='landsat_download'
DEFAULT_PROCESSED_PATH='landsat_processed'
REMOVE_AFTER_PROCESS=False
DOWNLOAD_BANDS=[2,3,4,7]
DOWNLOAD_ADDITIONAL_FILES=["_BQA.TIF", "_MTL.txt", "_ANG.txt"]
DOWNLOAD_BASE_URL='https://landsat-pds.s3.amazonaws.com/c1/L8'


parser = argparse.ArgumentParser()
parser.add_argument('--target_csv', default=FIRE_LABEL_FILE, help="Fire label file")
parser.add_argument('--download_dir', default=DEFAULT_DOWNLOAD_PATH, help="Directory with the SIGNS dataset")
parser.add_argument('--processed_dir', default=DEFAULT_PROCESSED_PATH, help="Where to write the new data")


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
        print "Download directory for {} already exists!".format(savepath)
    else:
        os.mkdir(savepath)
    print "[Downloading Image] : {}".format(download_key)

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

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.download_dir), "Couldn't find the dataset at {}".format(args.download_dir)

    with open(args.target_csv, 'r') as f:
        f.readline()
        # skip first line
        meta_fire = {}
        scene_list = []
        count = 0
        for line in f:
            (fid, area, _1, _2, fire_id, lat, lon, datetime, julian, gmt) = line.split(",")[:-5]
            lon = float(lon)
            lat = float(lat)
            date = datetime.split()[0].split("/")
            date = "20{0:0>2}-{1:0>2}-{2:0>2}".format(date[2], date[0], date[1])
            scene_id, download_key, cloud = get_image_ID(float(lon), float(lat), date)
            if scene_id != -1:
                if download_key not in meta_fire:
                    meta_fire[download_key] = {"lon":lon, "lat":lat, "date":date, "scene_id":scene_id, "cloud":cloud, "fires":[fire_id]}
                else:
                    meta_fire[download_key]["fires"].append(fire_id)
                if download_key not in scene_list:
                    scene_list.append(download_key)
                    count += 1
            else:
                print "fail to retrieve fire {} with {}".format(fire_id, (lon, lat, date))
            if count > 10:
                # put a cap on # image to download first
                break
        meta_name_base = os.path.basename(args.target_csv).strip(".csv")
        with open(os.path.join(args.download_dir,  meta_name_base +"_meta.pickle"), 'w') as pickle_f:
            pickle.dump(meta_fire, pickle_f)
        with open(os.path.join(args.download_dir,  meta_name_base + "_scene_lists.pickle"), 'w') as pickle_f:
            pickle.dump(scene_list, pickle_f)
        with open(os.path.join(args.download_dir, meta_name_base+"_meta.json"), 'w') as json_f:
            json.dump(meta_fire, json_f, indent=4)
        with open(os.path.join(args.download_dir, meta_name_base+"_scene_lists.json"), 'w') as json_f:
            json.dump(scene_list, json_f, indent=4)
        # now download all scene image
        #exit(1)
        print "====== Start downloading all {} images =====".format(len(scene_list))
        for download_key in scene_list:
            #pass
            download_scene(download_key, savepath_base=args.download_dir)
        print "====== Start preprocessing all {} images =====".format(len(scene_list))
        for download_key in scene_list:
            preprocess(download_key, src_path=os.path.join(args.download_dir, download_key),
                       dst_path=os.path.join(args.download_dir, download_key))





