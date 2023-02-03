install: venv
	. venv/bin/activate; python3 -m pip install --upgrade pip;pip3 install -Ur requirments.txt
venv:
	test -d venv || python3 -m venv venv

clean:
	rm -rf venv
	rm -rf MNIST
	rm -rf MNIST_JPGS
run1:
	. venv/bin/activate; python3 src/Classifier_1.py
run2:
	. venv/bin/activate; python3 src/Classifier_2.py
run3:
	. venv/bin/activate;python3 src/Classifier_3.py
run4: 
	. venv/bin/activate; python3 src/Classifier_4.py 
