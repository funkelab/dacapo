from setuptools import setup

setup(
        name='again',
        version='0.1',
        url='https://github.com/funkelab/again',
        author='Jan Funke',
        author_email='funkej@janelia.hhmi.org',
        license='MIT',
        packages=[
            'again',
            'again.analyze',
            'again.evaluate',
            'again.models',
            'again.optimizers',
            'again.tasks',
            'again.tasks.post_processors',
            'again.tasks.losses',
            'again.tasks.predictors',
            'again.tasks.data',
            'again.store',
            'again.gp',
        ]
)
