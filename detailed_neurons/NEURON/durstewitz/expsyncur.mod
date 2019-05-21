NEURON {
	POINT_PROCESS ExpSynCur
	RANGE tau, i
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
}

PARAMETER {
	tau = 0.1 (ms) <1e-9,1e9>
}

ASSIGNED {
	i (nA)
}

STATE {
	g (nA)
}

INITIAL {
	g=0
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	i = -g
}

DERIVATIVE state {
	g' = -g/tau
}

NET_RECEIVE(weight (nA)) {
	g = g + weight
}
