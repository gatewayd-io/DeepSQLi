train-dataset1:
	# Train the model with dataset1
	python train.py ../dataset/sqli_dataset1.csv ../sqli_model/1
	# Add warm-up to the model to load the model into memory after
	# starting the Serving API container
	python warm_up.py ../dataset/sqli_dataset1.csv ../sqli_model/1

train-dataset2:
	# Train the model with dataset2
	python train.py ../dataset/sqli_dataset2.csv ../sqli_model/2
	# Add warm-up to the model to load the model into memory after
	# starting the Serving API container
	python warm_up.py ../dataset/sqli_dataset2.csv ../sqli_model/2

train-dataset2-model3:
	# Train the model with dataset2
	poetry run python train_v3.py ../dataset/sqli_dataset2.csv ../sqli_model/3
	# Add warm-up to the model to load the model into memory after
	# starting the Serving API container
	poetry run python warm_up.py ../dataset/sqli_dataset2.csv ../sqli_model/3 true
