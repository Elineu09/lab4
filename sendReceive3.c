#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Status status;
    int id, p, i, rounds;
    long int size; // Tamanho da mensagem em bytes
    double secs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc != 3) {
        if (!id) printf("Uso: %s <n-rounds> <msg-size-bytes>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }

    rounds = atoi(argv[1]);
    size = atol(argv[2]);

    // Alocação do buffer para a mensagem
    char *buffer = (char *)malloc(size);
    if (buffer == NULL) {
        if (!id) printf("Erro ao alocar memória.\n");
        MPI_Finalize();
        exit(1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    secs = -MPI_Wtime();

    for (i = 0; i < rounds; i++) {
        if (!id) {
            // Processo 0 envia para o 1 e recebe do último (p-1)
            MPI_Send(buffer, size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(buffer, size, MPI_BYTE, p - 1, 0, MPI_COMM_WORLD, &status);
        } else {
            // Outros processos recebem do anterior e enviam para o próximo
            MPI_Recv(buffer, size, MPI_BYTE, id - 1, 0, MPI_COMM_WORLD, &status);
            MPI_Send(buffer, size, MPI_BYTE, (id + 1) % p, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    secs += MPI_Wtime();

    if (!id) {
        double avg_time_per_transfer = secs / (rounds * p);
        printf("--- Resultados para %ld bytes ---\n", size);
        printf("Tempo Total: %f s\n", secs);
        printf("Tempo médio por transferência: %f us\n", avg_time_per_transfer * 1e6);
        
        if (size > 1) {
            // Cálculo aproximado de Largura de Banda (Bytes/segundo)
            // BW = Tamanho / Tempo
            double bw = size / avg_time_per_transfer;
            printf("Largura de Banda estimada: %f MB/s\n", bw / (1024 * 1024));
        } else {
            printf("Latência estimada (L): %f us\n", avg_time_per_transfer * 1e6);
        }
    }

    free(buffer);
    MPI_Finalize();
    return 0;
}