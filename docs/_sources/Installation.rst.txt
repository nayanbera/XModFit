.. _Installation:

Installation
============
Follow the following instructions for installation:

1) Install Anaconda python (Python 3.8 and higher) for your operating system from `Anaconda website <https://www.anaconda.com/products/individual>`_
2) Open a Anaconda terminal the run these two commands::

    conda install pyqt pyqtgraph sqlalchemy scipy six matplotlib pandas
    pip install lmfit pyfai pylint periodictable mendeleev corner emcee tabulate xraydb python-docx

3) The installation can be done in two different ways:

    a) Easier and preferred way with `GIT <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_ already installed with Anaconda installation, run the following commands in a terminal with your own folder path::

        cd /home/mrinal/Download/
        git clone https://github.com/nayanbera/XModFit


       The method will create **XModFit** folder with all updated packages in installation folder (i.e. /home/mrinal/Download). The advantage of this method is that it is easy to upgrade the package later on. In order to upgrade, go to the folder named **XModFit** and run the following command::

            git pull

    b) Universal way which does not need GIT installation:
	    i) Open a web browser and go to the webpage : https://github.com/nayanbera/XModFit
	    ii) Click the green button named "Clone or download"
	    iii) Download the zip file
   	    iv) Extract the zip file into a folder
   	    v) In the Anaconda terminal go the the extracted folder::

   	            cd /home/mrinal/Download/XModFit-master

4) Run the command to run **XModFit**::

            python XModFit.py

