# PROYECT 1
## INSTALLATION OF PYSPARK

How to install pyspark: https://medium.com/tinghaochen/how-to-install-pyspark-locally-94501eefe421
## PRELIMINARY STEPS

1. First, check that you have java jdk was installed in your pc.
2. Then, check make is installed in your pc.
2. Into the project folder, create three folders in the root directory:
    - results/
        - origin/
        - transform/
    - graphics/
    - data/
3. Download the dataset: http://medata.gov.co/dataset/atenciones-en-urgencias.
4. Move the dataset urgencias.cvs to data folder.
## RUN APP

Open a terminal in your operating system, and locate yourself whete you have stored the project.

If you are in Linux OS, run the command:

~~~
make start-ubuntu FILE=main.py RESULT=results_test
~~~

If you are in Windows OS, run the command:

~~~
make start-windows FILE=main.py RESULT=results_test
~~~

NOTE:

In the FILE attribute you must write the file that you want to run, then in the attribute RESULT write the name of the file
where you would like save the results.

## REFERENCES

1. https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.SparkSession.builder.appName.html 
2. http://www.saludcapital.gov.co/documents/resolucion_3047_2008.pdf
3. https://gist.github.com/justincbagley/ec0a6334cc86e854715e459349ab1446
4. https://medium.com/tinghaochen/how-to-install-pyspark-locally-94501eefe421