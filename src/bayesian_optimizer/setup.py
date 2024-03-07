from setuptools import find_packages, setup

package_name = 'bayesian_optimizer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jonas',
    maintainer_email='ohnemujo@gmail.com',
    description='This package contains a bayesian_optimizer node that optimizes node parameters, e.g. MPC cost parameters.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'talker = bayesian_optimizer.bayesian_optimizer_node:main',
        ],
    },
)
