#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

// Protótipos
void my_Bcast(void* data, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
void my_Scatter(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
void my_Scatterv(void* sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype,
                 void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);

int main(int argc, char** argv) {
    int id, p;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (p < 2) {
        if (id == 0) printf("Use pelo menos 2 processos.\n");
        MPI_Finalize();
        return 1;
    }

    double start, local_time, global_time_manual, global_time_native;

    // =====================================================
    // TESTE 1: BCAST
    // =====================================================
    int bcast_data = (id == 0) ? 100 : 0;

    // Cronometrar Manual
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    my_Bcast(&bcast_data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_time = MPI_Wtime() - start;
    MPI_Reduce(&local_time, &global_time_manual, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Cronometrar Nativo
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    MPI_Bcast(&bcast_data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_time = MPI_Wtime() - start;
    MPI_Reduce(&local_time, &global_time_native, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (id == 0) printf("Bcast   | Manual: %f s | Native: %f s\n", global_time_manual, global_time_native);

    // =====================================================
    // TESTE 2: SCATTER
    // =====================================================
    int send_size = 1000;
    int *send_buf = NULL;
    int *recv_buf = malloc(send_size * sizeof(int));

    if (id == 0) {
        send_buf = malloc(p * send_size * sizeof(int));
        for (int i = 0; i < p * send_size; i++) send_buf[i] = i;
    }

    // Manual
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    my_Scatter(send_buf, send_size, MPI_INT, recv_buf, send_size, MPI_INT, 0, MPI_COMM_WORLD);
    local_time = MPI_Wtime() - start;
    MPI_Reduce(&local_time, &global_time_manual, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Nativo
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    MPI_Scatter(send_buf, send_size, MPI_INT, recv_buf, send_size, MPI_INT, 0, MPI_COMM_WORLD);
    local_time = MPI_Wtime() - start;
    MPI_Reduce(&local_time, &global_time_native, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (id == 0) printf("Scatter | Manual: %f s | Native: %f s\n", global_time_manual, global_time_native);

    // =====================================================
    // TESTE 3: SCATTERV
    // =====================================================
    int *sendcounts = NULL;
    int *displs = NULL;
    int *send_buf_v = NULL;
    int recv_count = id + 1;
    int *recv_buf_v = malloc(recv_count * sizeof(int));

    if (id == 0) {
        sendcounts = malloc(p * sizeof(int));
        displs = malloc(p * sizeof(int));
        int total = 0;
        for (int i = 0; i < p; i++) {
            sendcounts[i] = i + 1;
            displs[i] = total;
            total += sendcounts[i];
        }
        send_buf_v = malloc(total * sizeof(int));
        for (int i = 0; i < total; i++) send_buf_v[i] = i;
    }

    // Manual
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    my_Scatterv(send_buf_v, sendcounts, displs, MPI_INT, recv_buf_v, recv_count, MPI_INT, 0, MPI_COMM_WORLD);
    local_time = MPI_Wtime() - start;
    MPI_Reduce(&local_time, &global_time_manual, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Nativo
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    MPI_Scatterv(send_buf_v, sendcounts, displs, MPI_INT, recv_buf_v, recv_count, MPI_INT, 0, MPI_COMM_WORLD);
    local_time = MPI_Wtime() - start;
    MPI_Reduce(&local_time, &global_time_native, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (id == 0) printf("Scatterv| Manual: %f s | Native: %f s\n", global_time_manual, global_time_native);

    // CLEANUP
    free(recv_buf);
    free(recv_buf_v);
    if (id == 0) {
        free(send_buf);
        free(send_buf_v);
        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}

// IMPLEMENTAÇÕES MANUAIS (Corrigidas)

void my_Bcast(void* data, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    int id, p;
    MPI_Comm_rank(comm, &id);
    MPI_Comm_size(comm, &p);

    if (id == root) {
        for (int i = 0; i < p; i++)
            if (i != root) MPI_Send(data, count, datatype, i, 0, comm);
    } else {
        MPI_Recv(data, count, datatype, root, 0, comm, MPI_STATUS_IGNORE);
    }
}

void my_Scatter(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {
    int id, p, size;
    MPI_Comm_rank(comm, &id);
    MPI_Comm_size(comm, &p);
    MPI_Type_size(sendtype, &size);

    if (id == root) {
        for (int i = 0; i < p; i++) {
            char* ptr = (char*)sendbuf + i * sendcount * size;
            if (i == root) memcpy(recvbuf, ptr, sendcount * size);
            else MPI_Send(ptr, sendcount, sendtype, i, 0, comm);
        }
    } else {
        MPI_Recv(recvbuf, recvcount, recvtype, root, 0, comm, MPI_STATUS_IGNORE);
    }
}

void my_Scatterv(void* sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype,
                 void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {
    int id, p, size;
    MPI_Comm_rank(comm, &id);
    MPI_Comm_size(comm, &p);
    MPI_Type_size(sendtype, &size);

    if (id == root) {
        for (int i = 0; i < p; i++) { // Loop corrigido para percorrer todos os processos
            char* ptr = (char*)sendbuf + displs[i] * size;
            if (i == root) memcpy(recvbuf, ptr, sendcounts[i] * size);
            else MPI_Send(ptr, sendcounts[i], sendtype, i, 0, comm);
        }
    } else {
        MPI_Recv(recvbuf, recvcount, recvtype, root, 0, comm, MPI_STATUS_IGNORE);
    }
}