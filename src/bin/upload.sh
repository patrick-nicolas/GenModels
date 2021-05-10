scp -i ~/.ssh/alpha-spark.pem *.py ec2-user@10.5.240.16:~/python/
scp -i ~/.ssh/alpha-spark.pem util/*.py ec2-user@10.5.240.16:~/python/util/
scp -i ~/.ssh/alpha-spark.pem nnet/*.py ec2-user@10.5.240.16:~/python/nnet/
scp -i ~/.ssh/alpha-spark.pem -r conf ec2-user@10.5.240.16:~/python/
scp -i ~/.ssh/alpha-spark.pem autoencoder/*.py ec2-user@10.5.240.16:~/python/autoencoder/
scp -i ~/.ssh/alpha-spark.pem datasets/*.py ec2-user@10.5.240.16:~/python/datasets/
scp -i ~/.ssh/alpha-spark.pem features/*.py ec2-user@10.5.240.16:~/python/features/
scp -i ~/.ssh/alpha-spark.pem embeddings/*.py ec2-user@10.5.240.16:~/python/embeddings/
