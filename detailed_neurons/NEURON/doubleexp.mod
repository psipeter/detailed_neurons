TITLE doubleexp synapse 

NEURON {
    POINT_PROCESS doubleexp
    NONSPECIFIC_CURRENT i
    RANGE g, a, b, e, tauRise, tauFall
}

UNITS {
    (uS) = (microsiemens)
    (nA) = (nanoamp)
    (mV) = (millivolt)
}

PARAMETER {
    tauRise = 0.1  (ms)
    tauFall = 10.0  (ms)
    e = 0  (mV)
}

ASSIGNED {
    v  (mV)
    i  (nA)
    g  (uS)
    factor
}

INITIAL { 
    a = 0  
    b = 0 
    factor = tauRise*tauFall / (tauFall-tauRise)
}

STATE {
    a (uS)
    b (uS)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = (b-a)*factor
    i = g*(v-e)
}

DERIVATIVE states {
    a' = -a/tauRise
    b' = -b/tauFall
}

NET_RECEIVE(weight (uS)) {
    a = a + weight
    b = b + weight
}