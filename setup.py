from distutils.core import setup
setup(
  name = 'classify_animals',
  packages = ['classify_animals', 'classify_animals.scripts'],  
  version = '1.0',  
  license ='CC BY 4.0', 
  description = """Provides a function that classifies images 
      of Hong Kong animals that have been 
      detected by Megadetector by species""",  
  author = 'Gareth Lamb', 
  author_email = 'lambg@hku.hk',
  url = 'https://github.com/garethlamb/animal_classifier_pipeline',
  keywords = [
    'Animal', 
    'Classifier', 
    'Pipeline', 
    'Classification', 
    'Identifier',
    'Species',
    'Hong Kong'
  ],
  install_requires=[   
    'pandas',
    'numpy',
    'Pillow',
    'scikit-learn',
    'keras',
    'tensorflow',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Researchers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: CC BY 4.0',
    'Programming Language :: Python :: 3.11.3', 
  ],
)