# torchpack dist-run -np 1 python3 tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox --out result.pkl
torchpack dist-run -np 1 python3 tools/simulator.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --settings-path /airsim/settings.json \
  --mode pred \
  --bbox-score 0.5 \
  --checkpoint pretrained/bevfusion-det.pth \
  --out-dir ./vis_results