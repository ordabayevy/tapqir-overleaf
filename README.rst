Analysis output of experimental data is stored in ``experimental`` folder:

* DatasetA: ``experimental/DatasetA``
* DatasetB: ``experimental/DatasetB``
* DatasetC: ``experimental/DatasetC``
* DatasetD: ``experimental/DatasetD``
* DatasetA: ``experimental/P10DatasetA`` (10x10 AOIs)
* DatasetA: ``experimental/P6DatasetA`` (6x6 AOIs)

Analysis output of simulated data is stored in ``simulations`` folder:

* Supplemental Data 1: ``simulations/lamda*``
* Supplemental Data 2: ``simulations/seed*``
* Supplemental Data 3: ``simulations/height*``
* Supplemental Data 4: ``simulations/negative*``
* Supplemental Data 5: ``simulations/kon*``
* Supplemental Data 6: ``simulations/sigma*``

These folder contain following analysis outputs:

* ``data.tpqr`` AOI images
* ``cosmos-channel0-params.tpqr``: posterior parameter distributions
* ``cosmos-channel0-summary.csv``: summary of global parameter values

Figure 1
--------

.. image:: figures/cosmos_experiment/cosmos_experiment.png
   :caption: Example CoSMoS experiment.

Image file: ``figures/cosmos_epxeriment/cosmos_experiment.png``


Figure 2
--------

.. image:: figures/graphical_model.png
   :caption: Depiction of the cosmos probabilistic image model and model parameters.

Image file: ``figures/graphical_model.png``

To generate panels A, B, and C in the image, run (outpus ``figures/graphical_model.svg`` vector image)::

  python scripts/figures/graphical_model.py

Input data:

* ``experimental/DatasetA``

Graphical model in panel D is located at ``figures/graphical_model.pdf``.

Figure 2–Figure supplement 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: figures/graphical_model_extended.png
   :caption: Extended graphical representation of the cosmos generative probabilistic model.

Image file: ``figures/graphical_model_extended.png``

Figure 2–Figure supplement 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: figures/graphical_model_xy.png
   :caption: The prior distributions for x and y spot position parameters.

Image file: ``figures/graphical_model_xy.png``

To generate the image file, run::

  python scripts/figures/graphical_model_xy.py


Figure 3
--------

.. image:: figures/tapqir_analysis.png
   :caption: Tapqir analysis and inferred model parameters.

Image file: ``figures/tapqir_analysis.png``

To generate the image file, run::

  python scripts/figures/tapqir_analysis.py

Input data:

* ``simulations/lamda0.5`` (panel A)
* ``experimental/DatasetA`` (panel B)

Figure 3-Figure supplement 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: figures/tapqir_analysis_probs.png
   :caption: Calculated spot probabilities.

Image file: ``figures/tapqir_analysis_probs.png``

To generate the image file, run::

  python scripts/figures/tapqir_analysis_probs.py

Input data:

* ``simulations/lamda0.5`` (panel A)
* ``experimental/DatasetA`` (panel B)

Figure 3-Figure supplement 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: figures/tapqir_analysis_ppc.png
   :caption: Reproduction of experimental data by posterior predictive sampling.

Image file: ``figures/tapqir_analysis_ppc.png``

To generate the image file, run::

  python scripts/figures/tapqir_analysis_ppc.py

Input data:

* ``experimental/DatasetA`` (panel A)
* ``experimental/DatasetB`` (panel B)
* ``experimental/DatasetC`` (panel C)
* ``experimental/DatasetD`` (panel D)

Figure 3-Figure supplement 3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: figures/tapqir_analysis_randomized.png
   :caption: Tapqir analysis of image data simulated using a broad range of global parameters.

Image file: ``figures/tapqir_analysis_randomized.png``

To generate the image file, run::

  python scripts/figures/tapqir_analysis_randomized.py

Input data:

* ``simulations/seed{0-16}``

Figure 3-Figure supplement 4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: figures/tapqir_analysis_size.png
   :caption: Effect of AOI size on analysis of experimental data.

Image file: ``figures/tapqir_analysis_size.png``

To generate the image file, run::

  python scripts/figures/tapqir_analysis_size.py

Input data:

* ``experimental/DatasetA`` (14x14 AOIs)
* ``experimental/P10DatasetA`` (10x10 AOIs)
* ``experimental/P6DatasetA`` (6x6 AOIs)


Figure 4
--------

Tapqir performance on simulated data with different SNRs or different non-specific binding rates.

To generate source image file ``figures/tapqir_performance.png``, run::

  python scripts/figures/tapqir_performance.py

Input data:

* ``simulations/height*`` (panels A, B, C, D)
* ``simulations/lamda*`` (panels E, F, G, H)
* ``simulations/negative*`` (panel I)

Figure 4-Figure supplement 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

False negative spot misidentifications by Tapqir and spot-picker method.

To generate source image file ``figures/tapqir_performance_fn.png``, run::

  python scripts/figures/tapqir_performance_fn.py

Input data:

* ``simulations/lamda1``
* ``simulations/spotpicker_result.mat`` (spot-picker analysis output)


Figure 5
--------

Tapqir analysis of association/dissociation kinetics and thermodynamics.

To generate source image file ``figures/kinetic_analysis.png``, run::

  python scripts/figures/kinetic_analysis.py

Input data:

* ``simulations/kon0.01lamda0.01``
* ``simulations/kon0.01lamda0.15``
* ``simulations/kon0.01lamda0.5``
* ``simulations/kon0.01lamda1``
* ``simulations/kon0.02lamda0.01``
* ``simulations/kon0.02lamda0.15``
* ``simulations/kon0.02lamda0.5``
* ``simulations/kon0.02lamda1``
* ``simulations/kon0.03lamda0.01``
* ``simulations/kon0.03lamda0.15``
* ``simulations/kon0.03lamda0.5``
* ``simulations/kon0.03lamda1``


Figure 6
--------

Extraction of target-binder association kinetics from example experimental data.

To generate source image file ``figures/experimental_data.png``, run::

  python scripts/figures/DatasetB_ttfb_analysis.py
  python scripts/figures/experimental_data.py

Input data:

* ``experimental/DatsetB``

Figure 6-Figure supplement 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additional example showing extraction of target-binder association kinetics from experimental data.

To generate source image file ``figures/experimental_data_DatasetA.png``, run::

  python scripts/figures/DatasetA_ttfb_analysis.py
  python scripts/figures/experimental_data_DatasetA.py

Input data:

* ``experimental/DatsetA``

Figure 6-Figure supplement 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additional example showing extraction of target-binder association kinetics from experimental data.

To generate source image file ``figures/experimental_data_DatasetC.png``, run::

  python scripts/figures/DatasetC_ttfb_analysis.py
  python scripts/figures/experimental_data_DatasetC.py

Input data:

* ``experimental/DatsetC``

Figure 6-Figure supplement 3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additional example showing extraction of target-binder association kinetics from experimental data.

To generate source image file ``figures/experimental_data_DatasetD.png``, run::

  python scripts/figures/DatasetD_ttfb_analysis.py
  python scripts/figures/experimental_data_DatasetD.py

Input data:

* ``experimental/DatsetD``


Supplemental Data 1
-------------------

Varying non-specific binding rate simulation parameters and corresponding fit values

To generate source image file ``supplementary/data1.xlsx``, run::

  python scripts/supplementary/data1.py

Input data:

* ``simulations/lamda*``


Supplemental Data 2
-------------------

Randomized simulation parameters and corresponding fit values

To generate source image file ``supplementary/data2.xlsx``, run::

  python scripts/supplementary/data2.py

Input data:

* ``simulations/seed*``


Supplemental Data 3
-------------------

Randomized simulation parameters and corresponding fit values

To generate source image file ``supplementary/data3.xlsx``, run::

  python scripts/supplementary/data3.py

Input data:

* ``simulations/height*``


Supplemental Data 4
-------------------

No target-specific binding and varying non-specific binding rate simulation parameters and corresponding fit values

To generate source image file ``supplementary/data4.xlsx``, run::

  python scripts/supplementary/data4.py

Input data:

* ``simulations/negative*``


Supplemental Data 5
-------------------

Kinetic simulation parameters and corresponding fit values

To generate source image file ``supplementary/data5.xlsx``, run::

  python scripts/supplementary/data5.py

Input data:

* ``simulations/kon*``


Supplemental Data 6
-------------------

Varying proximity simulation parameters and corresponding fit values

To generate source image file ``supplementary/data6.xlsx``, run::

  python scripts/supplementary/data6.py

Input data:

* ``simulations/sigma*``