: Persistent Na+ channel with Dopamine Perturbation

NEURON {
	SUFFIX Nap_DA
	USEION na READ ena WRITE ina
	RANGE gnapbar, ina, gna
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	
}

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

PARAMETER {
	v (mV)
	dt (ms)
	gnapbar= 0.0022 (mho/cm2) <0,1e9>
	ena = 55 (mV)
}

STATE {
	m h
}

ASSIGNED {
	ina (mA/cm2)
	minf hinf 
	mtau (ms)
	htau (ms)
	gna (mho/cm2)
	
}

INITIAL {
	rate(v)
	m = minf
	h = hinf
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gna = gnapbar*m*h
	ina = gna*(v-55)
	
}

DERIVATIVE states {
	rate(v)
	m' = (minf-m)/mtau
	h' = (hinf-h)/htau
}

UNITSOFF

FUNCTION malf( v){ LOCAL va 
	va=v+17
	if (fabs(va)<1e-04){
	 va = va + 0.00001 }
	malf = (-0.2816*va)/(-1+exp(-va/9.3))
	
}


FUNCTION mbet(v(mV))(/ms) { LOCAL vb 
	vb=v-10
	if (fabs(vb)<1e-04){
	    vb = vb + 0.00001 }

	mbet = (0.2464*vb)/(-1+exp(vb/6))

}	


FUNCTION half(v(mV))(/ms) { LOCAL vc 
	vc=v+42.8477
	if (fabs(vc)<1e-04){
	   vc=vc+0.00001 }
        half= (2.0e-5)*(exp(-vc/4.0248))

}


FUNCTION hbet(v(mV))(/ms) { LOCAL vd
	vd=v-413.9284
	if (fabs(vd)<1e-04){
	vd=vd+0.00001 }
        hbet= (0.014286)/(1+exp(-vd/148.2589))
 
}




PROCEDURE rate(v (mV)) {LOCAL msum, hsum, ma, mb, ha, hb
	ma=malf(v) mb=mbet(v) ha=half(v) hb=hbet(v)
	
	msum = ma+mb
	minf = ma/msum
	mtau = 1/msum
	
	
	hsum = ha+hb
	hinf = ha/hsum
	htau = 1/hsum
}

	
UNITSON




