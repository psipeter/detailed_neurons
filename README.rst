***********************************************************
detailed_neurons: build nengo networks with complex neurons
***********************************************************

Includes methods for training encoders, decoders, and filters in nengo networks containing detailed neuron models.

Test the performance of these methods on several dynamical systems, including: communication channel, f(x)=x^2, integrator, 2d oscillator, Lorenz attractor


Clone the repository
=======
	
	git clone https://github.com/psipeter/detailed_neurons.git

	cd detailed_neurons
    
Install virtual environment
=======

	pip3 install pipenv

    pipenv shell


Install NEURON
=======
    
	(follow instructions at https://neuron.yale.edu/neuron/download/compile_linux)

	cd detailed_neurons/detailed_neurons/NEURON

	module load mpi

	/pwd/x86_64/bin/nrnivmodl

	(pwd could be /usr/local/nrn/, or an install directory in your home or user folder)


Install detailed_neurons
=======

	cd ../..
    
	pipenv --three install '-e .'
