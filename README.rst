***********************************************************
detailed_neurons: build nengo networks with complex neurons
***********************************************************

Includes methods for training encoders, decoders, and filters in nengo networks containing detailed neuron models.

Test the performance of these methods on several dynamical systems, including: communication channel, f(x)=x^2, integrator, 2d oscillator, Lorenz attractor


Install
=======

	git clone https://github.com/psipeter/detailed_neurons.git

	cd detailed_neurons
    
	pip install pipenv

    pipenv --three shell
    
    wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.6/nrn-7.6.tar.gz
    
    nrn-7.6.tar.gz
    
    cd nrn-7.6

    ./configure --with-nrnpython --with-pyexe=python3 --without-iv

    make

    sudo make install

    cd src/nrnpython

    python3 setup.py install

    cd NEURON/durstewitz/

    /usr/local/nrn/x86_64/bin/nrnivmodl
    
    cd ../../..
    
    (edit path in neuron_models.py and train.py to allow loading of durstewitz.hoc)
    
	pipenv --three install '-e .'