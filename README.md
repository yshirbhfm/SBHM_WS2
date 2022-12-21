# SBHMFitPy
 
$$ \nabla_b=-iC_{grad}\hat{k}_{in} $$
- C_grad: a fitting parameter related to the magnitude of the gradient
  - Since a phase shift between the interface and bulk contributions cannot be excluded Cgrad was taken as a complex number.
 
 Fresenel of layers to apply for contributions
 | Fresenel | SH Contribution |
 | :---: | :---: |
 | n0 | air |
 | n1 | interface 1 |
 | n1 | bulk 1 |
 | n2 | interface 2 |
 | n2 | bulk 2 |
 
 [1] C.Reitböck, D.Stifter, A.Alejo-Molina, K.Hingerl, andH.Hardhienata, “Bulk quadrupole and interface dipole contribution for second harmonic generation in Si(111),” J. Opt., vol. 18, no. 3, p. 035501, Mar.2016, doi: (10.1088/2040-8978/18/3/035501.)[https://iopscience.iop.org/article/10.1088/2040-8978/18/3/035501]

# SHGsim
## simplified bond hyperpolarizability model
read the following for the basic algorithm
1. C.Reitböck, D.Stifter, A.Alejo-Molina, K.Hingerl, andH.Hardhienata, “Bulk quadrupole and interface dipole contribution for second harmonic generation in Si(111),” J. Opt., vol. 18, no. 3, p. 035501, Mar.2016, doi: 10.1088/2040-8978/18/3/035501.

## Schematics of Corresponding optical setup
![image](https://user-images.githubusercontent.com/61143470/185018593-2700ee79-b943-46c5-8af1-749f4f404b8a.png)
 
## Contributions
### Surface dipole [1]
![image](https://user-images.githubusercontent.com/61143470/184480841-5feaaa8a-df85-49b5-b62a-4b8a257c8fbd.png)
### Bulk Quadrupole [1]
![image](https://user-images.githubusercontent.com/61143470/184480854-5b93342b-633d-401b-9a14-0fbe49db935f.png)
### Bulk Dipole (non-centrosymmetric structure only)
1. H.Hardhienata, A.Alejo-Molina, C.Reitböck, A.Prylepa, D.Stifter, andK.Hingerl, “Bulk dipolar contribution to second-harmonic generation in zincblende,” J. Opt. Soc. Am. B, vol. 33, no. 2, p. 195, Feb.2016, doi: 10.1364/JOSAB.33.000195.

![image](https://user-images.githubusercontent.com/61143470/184482077-fd2d33f2-33e4-4970-858c-cc6846fb0db7.png)

where $\alpha_{GaAs} = \alpha_{Ga} - \alpha_{As}$
 
## bond structure presets
### Surface
#### [tetrahedral] Si(111) 1st layer & ZnO(002) unreduced surface
β = 109.47° 

<img src='https://user-images.githubusercontent.com/61143470/134481247-f143e54b-de5f-4a0e-9ef9-f08249798173.png' width=300/> <img src='https://user-images.githubusercontent.com/61143470/134481349-071b5101-03dd-44e4-bc18-d5059dd67006.png' width=200/>
1. Reitböck, C., Stifter, D., Alejo-Molina, A., Hingerl, K. &Hardhienata, H. Bulk quadrupole and interface dipole contribution for second harmonic generation in Si(111). J. Opt. 18, 035501 (2016).
2. Hardhienata, H., Priyadi, I., Alatas, H., Birowosuto, M. D. &Coquet, P. Bond model of second-harmonic generation in wurtzite ZnO(0002) structures with twin boundaries. _J. Opt. Soc. Am. B_ **36**, 1127 (2019).

#### Si(100) 1~2nd layer
<img src='https://user-images.githubusercontent.com/61143470/135051124-7593ce6f-a8d3-43b1-8a0f-09757fe34be1.png' width=800/> ![image](https://user-images.githubusercontent.com/61143470/135050344-0039dec7-cc2d-43e3-be14-68b0d7369a29.png)
1. Sipe, J., Moss, D. &vanDriel, H. Phenomenological theory of optical second- and third-harmonic generation from cubic centrosymmetric crystals. Phys. Rev. B 35, 1129–1141 (1987).

#### ZnO twin boundary (0002)
![image](https://user-images.githubusercontent.com/61143470/142721934-34f11022-72aa-4ac0-b8ab-ba00b8a623be.png) ![image](https://user-images.githubusercontent.com/61143470/142721918-216fc456-1979-4fc9-96bc-7f13a7a990ba.png)

1. Hardhienata, H., Priyadi, I., Alatas, H., Birowosuto, M. D. &Coquet, P. Bond model of second-harmonic generation in wurtzite ZnO(0002) structures with twin boundaries. _J. Opt. Soc. Am. B_ **36**, 1127 (2019).

#### [cubic] ZnO2(001) surface (mp-8484)
octahedron with 2 kinds of axis direction in the same height

![image](https://user-images.githubusercontent.com/61143470/142722782-4d34dd12-357c-49fc-9b29-160c0c6d938d.png)

Zn |   | 1 |   |   | 2 |  
-- | :--: | :--: | :--: | :--: | :--: | :--:
  | x | y | z | x | y | z
b1 | 0.0874 | -0.0874 | 0.4126 | -0.4126 | 0.4126 | 0.4126
b2 | 0.0874 | -0.4126 | -0.0874 | -0.4126 | 0.0874 | -0.0874
b3 | 0.4126 | 0.0874 | -0.0874 | -0.0874 | -0.4126 | -0.0874
b4 | 0.4126 | 0.4126 | 0.4126 | -0.0874 | -0.0874 | 0.4126
b5 | -0.4126 | -0.4126 | -0.4126 | 0.0874 | 0.0874 | -0.4126
b6 | -0.4126 | -0.0874 | 0.0874 | 0.0874 | 0.4126 | 0.0874
b7 | -0.0874 | 0.4126 | 0.0874 | 0.4126 | -0.0874 | 0.0874
b8 | -0.0874 | 0.0874 | -0.4126 | 0.4126 | -0.4126 | -0.4126

1. [material project mp-8484: ZnO2 (cubic, Pa-3, 205)](https://materialsproject.org/materials/mp-8484/#)

### Bulk
#### ZnO(0002) tetrahedral
β = 109.47° 

![image](https://user-images.githubusercontent.com/61143470/135049246-02a344ee-c481-4769-ade8-358c9280ec81.png) ![image](https://user-images.githubusercontent.com/61143470/135049429-f9338d0a-431f-4d61-9228-809e5d20f21b.png)
1. Hardhienata, H., Priyadi, I., Alatas, H., Birowosuto, M. D. &Coquet, P. Bond model of second-harmonic generation in wurtzite ZnO(0002) structures with twin boundaries. _J. Opt. Soc. Am. B_ **36**, 1127 (2019).
