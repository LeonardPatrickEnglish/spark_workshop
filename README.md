# Practical PySpark Workshop 


#### Download Dataset
Download the dataset (1.8 GB) from https://data.sfgov.org/Public-Safety/Fire-Department-Calls-for-Service/nuek-vuh3.


#### Setup Spark on Ubuntu

###### Install Scala and Java
```
cd ~
sudo apt install default-jre scala


wget https://www-us.apache.org/dist/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz
tar xvf spark-2.4.0-bin-hadoop2.7.tgz
sudo mv spark-2.4.0-bin-hadoop2.7 /usr/local/spark
```

##### Put these lines in .bashrc
```
export SPARK_HOME=/usr/local/spark
export PATH=$PATH:$SPARK_HOME/bin
export JAVA_HOME=/usr/lib/jvm/default-java
```

##### Refresh .bashrc file and Test Pyspark
```
source .bashrc
pyspark
```



#### Setup Cassandra

###### Install Java
```
sudo add-apt-repository ppa:webupd8team/java
sudo apt update; sudo apt install oracle-java8-installer -y
sudo apt install oracle-java8-set-default 
```

###### Install cassandra
```
echo "deb http://www.apache.org/dist/cassandra/debian 311x main" | sudo tee -a /etc/apt/sources.list.d/cassandra.sources.list
curl https://www.apache.org/dist/cassandra/KEYS | sudo apt-key add -
sudo apt-get update ; sudo apt-get install cassandra -y
```

###### start cassandra -  takes 1-2 min to start
```
sudo service cassandra start
```

###### check status - It should say active
```
sudo service cassandra status
```

#### Setup AWS free tier account with IAM roles and Access/Secret Keys
Note:  The AWS EMR tutorial will cost you ~2 Euro/hour
