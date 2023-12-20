
GARF = GATE ARF

```pip install garf```

Scripts associated with the publication:
Phys Med Biol. 2018 Oct 17;63(20):205013. doi: 10.1088/1361-6560/aae331.
Learning SPECT detector angular response function with neural network for accelerating Monte-Carlo simulations.
Sarrut D, Krah N, Badel JN, Létang JM.
https://www.ncbi.nlm.nih.gov/pubmed/30238925

A method to speed up Monte-Carlo simulations of single photon emission computed tomography (SPECT) imaging is proposed. It uses an artificial neural network (ANN) to learn the angular response function (ARF) of a collimator-detector system. The ANN is trained once from a complete simulation including the complete detector head with collimator, crystal, and digitization process. In the simulation, particle tracking inside the SPECT head is replaced by a plane. Photons are stopped at the plane, and the energy and direction are used as input to the ANN, which provides detection probabilities in each energy window. Compared to histogram-based ARF, the proposed method is less dependent on the statistics of the training data, provides similar simulation efficiency, and requires less training data. The implementation is available within the GATE platform.

- Gate10 test: https://github.com/OpenGATE/opengate/blob/master/opengate/tests/src/test043_garf.py
