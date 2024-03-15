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
