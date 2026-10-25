"""Exact k=2 coherent moments with complex overlap-ratio labels.

For a=<g|u>, b=<g|v>, A_u(v)=d2 integral |a|^2 b g and
B_u(v)=d2 integral a^2 b* g. These correspond to the canonical synthesis
integrand multiplied by b/a and its conjugate, respectively. The labels are
complex and generally unbounded, not positive probabilities or atom subsets.
"""


def coherent_parts(replica, witness, vector):
    derivative=replica.derivative(witness,vector)
    imaginary=replica.derivative(witness,1j*vector)
    ordinary=(derivative-1j*imaginary)/4
    conjugate=(derivative+1j*imaginary)/2
    return {'complex_ratio':ordinary,'conjugate_ratio':conjugate,
            'real_ratio':(ordinary+conjugate)/2,
            'imaginary_ratio':(ordinary-conjugate)/(2j)}
