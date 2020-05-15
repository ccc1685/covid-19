// time transition functions for beta and sigmac
  real transition(real base,real location,real t) {
          real scale;
          if (base == 1)
              scale = 1;
          else
             scale = base + (1. - base)/(1. + exp(.2*(t - location)));
          return scale;
  }
