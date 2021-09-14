from distutils.core import setup
setup(
    name='aituring_pipeline_efficientdet',         # How you named your package folder (MyLib)
    packages=['aituring_pipeline_efficientdet'],   # Chose the same as "name"
    version='0.1',      # Start with a small number and increase it with every change you make
    license='Private software',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description='AITuring EfficientDet Repositorio Refactorizado',   # Give a short description about your library
    author='Daniel Tobon Collazos',                   # Type in your name
    author_email='daniel.tobon@aituring.co',      # Type in your E-Mail
    url='https://github.com/danielTobon43',   # Provide either the link to your github or to your website
    download_url='https://git-codecommit.us-east-1.amazonaws.com/v1/repos/aituring_pipeline_efficientdet',    # I explain this later on
    keywords=['efficientdet', 'tf1', 'aituring'],   # Keywords that define your package best
    install_requires=[            # I get to this in a second
        'absl-py==0.13.0',
        'configparser==5.0.2',
        'lxml==4.6.3',
        'Pillow==8.3.2',
        'pycocotools==2.0.2; sys_platform=="linux"',
        'pycocotools-windows==2.0.0.2; sys_platform=="win32"',
        'opencv-python==4.5; sys_platform=="win32"',
        'opencv==4.5.3; sys_platform=="linux"',
        'PyYAML==5.4.1',
        'tensorboard==2.5.0',
        'tensorboard-data-server==0.6.0',
        'tensorboard-plugin-wit==1.8.0',
        'tensorflow==2.5.0',
        'tensorflow-estimator==2.5.0',
        'tensorflow-model-optimization==0.5.1.dev0'
    ],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3.8'
    ],
)
