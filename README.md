# Encore-ICNP

##### code structure

```
data/ the private dataset from bank
encore/ core model code and train code
evaluation/ evaluation code
	generate_traffic.ipynb synthesis traffic
	model_analysis.ipynb reproduce most result fig
figures/ 
preprocess/ preprocess dataset
	traffic_analysis.ipynb reproduce traffic analysis fig
results/
	generated traffic
simulation/
	ns3-based simulation for traffic verfication
utils/
```

##### reproduce

**dependency in requirements.txt**

first, run ./preprocess/process_raw.ipynb and ./preprocess/traffic_analysis.ipynb to get traffic analysis result

then, run ./encore/train_model.py to train Encore model

next, run ./evaluation/test_models.py and ./evaluation/model_analysis.ipynb to get systhetic traffic and accuracy&coverage 

for downstream task using traffic demand, run ./evaluation/generate_config.ipynb and ./evaluation/generate_traffic.ipynb, then go to sub-dir ./simulation to use ns-3 based simulator to run verfication using traffic demand, then run ./evaluation/traffic_analysis.ipynb to analysis the impact of different traffic demand on downstream tasks.

##### dataset

processed dataset in sub-dir ./data

full raw dataset available at [https://github.com/ruixu221/Tardis-FinApps-Dataset](https://github.com/ruixu221/Tardis-FinApps-Dataset)
