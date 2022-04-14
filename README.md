# quure_medical_devices_ocr

# run

to run install python = 3.8

pip install pymongo

start mongo
	
	sudo systemctl start mongod

step-1 clone the repositry
	
	git clone <https://github.com/usman-250/quure_medical_devices_ocr.git>

step-2 
	
	cd <path to repo>

step-3 make a virtual env
        
      python3 -m venv medical
  
step-4 activate env
      
	cd <path to venv/bin >
      
	source activate

step-5 install requirements
  
	pip install -r <path to requirements.txt file>
  
  
step-6

	export FLASK_APP = <path to root dir of cloned repo>

 	flask run

