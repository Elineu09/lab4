#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ARRAY_SIZE 1000000 // 1 milhão de inteiros (~4MB)

int main(int argc, char *argv[]) {
    int id, p;
    int *data = (int *)malloc(ARRAY_SIZE * sizeof(int));
    double start_time, end_time_indiv, end_time_bcast;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // Inicialização dos dados no Processo 0
    if (id == 0) {
        for (int i = 0; i < ARRAY_SIZE; i++) data[i] = i;
    }

    // --- TESTE 1: Envios Individuais (Ponto-a-Ponto) ---
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    if (id == 0) {
        for (int target = 1; target < p; target++) {
            MPI_Send(data, ARRAY_SIZE, MPI_INT, target, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(data, ARRAY_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    end_time_indiv = MPI_Wtime() - start_time;

    // --- TESTE 2: MPI_Bcast (Coletivo) ---
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    MPI_Bcast(data, ARRAY_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    end_time_bcast = MPI_Wtime() - start_time;

    // --- Relatório de Desempenho ---
    if (id == 0) {
        printf("\n--- Comparação de Desempenho (p=%d) ---\n", p);
        printf("Tempo Envios Individuais: %f s\n", end_time_indiv);
        printf("Tempo MPI_Bcast:          %f s\n", end_time_bcast);
        printf("Melhoria:                 %.2fx\n", end_time_indiv / end_time_bcast);
    }

    free(data);
    MPI_Finalize();
    return 0;
}