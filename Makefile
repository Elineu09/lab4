sendReceive: sendReceive.c
	mpicc sendReceive.c -o sendReceive

sendReceive3: sendReceive3.c
	mpicc sendReceive3.c -o sendReceive3

sendReceive4: sendReceive4.c
	mpicc sendReceive4.c -o sendReceive4

challenge: challenge.c
	mpicc challenge.c -o challenge

clean: 
	rm -f sendReceive 
