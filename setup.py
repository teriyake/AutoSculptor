from setuptools import setup, find_packages

setup(
	name="autosculptor",
	version="0.1.0",
	packages=find_packages(),
	install_requires=[
		"numpy>=1.20.0",
		"PyQt5>=5.15.0",
		"pytest>=7.0.0",
		"potpourri3d>=1.2.1",
	],
	extras_require={
		"dev": [
			"pytest>=7.0.0",
			"black-with-tabs>=22.10.0",
			"flake8>=4.0.1",
			"sphinx>=4.4.0",
		],
	},
	python_requires=">=3.10.8",
)
