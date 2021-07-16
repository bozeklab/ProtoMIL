#!/usr/bin/env bash

tsp -S 4

for i in $(seq -f "%03g" 1 160); do
  echo normal $i
  rclone copy gdrive:/training/normal/normal_$i.tif --drive-root-folder-id 0BzsdkU4jWx9Bb19WNndQTlUwb2M ../data/CAMELYON/ -P
  tsp bash -c "python camelyon_preproc.py ../data/CAMELYON/normal_$i.tif; rm -rf ../data/CAMELYON/normal_$i.tif;"
done

for i in $(seq -f "%03g" 1 111); do
  echo tumor $i
  rclone copy gdrive:/training/tumor/tumor_$i.tif --drive-root-folder-id 0BzsdkU4jWx9Bb19WNndQTlUwb2M ../data/CAMELYON/ -P
  tsp bash -c "python camelyon_preproc.py ../data/CAMELYON/tumor_$i.tif; rm -rf ../data/CAMELYON/tumor_$i.tif;"
done

for i in $(seq -f "%03g" 46 130); do
  echo test $i
  rclone copy gdrive:/testing/images/test_$i.tif --drive-root-folder-id 0BzsdkU4jWx9Bb19WNndQTlUwb2M ../data/CAMELYON/ -P
  tsp bash -c "python camelyon_preproc.py ../data/CAMELYON/test_$i.tif; rm -rf ../data/CAMELYON/test_$i.tif;"
done
