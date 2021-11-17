/* Grupo 6 */


#include <stdio.h>
#include <stdlib.h>
#include <glpk.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>

#define EPSILON 0.000001

#ifdef DEBUG
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif

typedef struct{
  int num; /* numero do item */
  double valor; /* valor do item */
  int peso; /* peso do item */
} Titem;

typedef struct{
  int n; /* total de itens */
  Titem *item; /* conjunto dos itens */
  int k; /* total de mochilas */
  int *C; /* capacidade das mochilas */
} Tinstance;
typedef struct{
  glp_prob *mip;
  int nodes;
  int ativos;
  double best_dualBound;
  double best_primalBound;
  double gap;
} my_infoT;

int carga_lp(glp_prob **lp, Tinstance I);
int carga_instancia(char* filename, Tinstance *I);
void  free_instancia(Tinstance I);
int RandomInteger(int low, int high);
double heuristica(Tinstance I, int tipo, char* entrada, double* x);
double otimiza_PLI(Tinstance I, int tipo, double* x, my_infoT* info);
void my_callback (glp_tree *tree, void *infop);

/* carrega o modelo de PLI nas estruturas do GLPK */
int carga_lp(glp_prob **lp, Tinstance I)
{
  int *ia, *ja, nrows, ncols, i, k, row, col, nz;
  double *ar;
  char name[80];// nome da restricao

  nrows=I.k+I.n; // 1 restricao de capacidade para cada mochila + 1 para cada item
  ncols=I.n*I.k;

  // Aloca matriz de coeficientes (o tamanho deve ser 2*n*k)
  ia=(int*)malloc(sizeof(int)*(I.n*I.k*2+1));
  ja=(int*)malloc(sizeof(int)*(I.n*I.k*2+1));
  ar=(double*)malloc(sizeof(double)*(I.n*I.k*2+1));

  // Cria problema de PL
  *lp = glp_create_prob();
  glp_set_prob_name(*lp, "mochila_multipla");
  glp_set_obj_dir(*lp, GLP_MAX);

  // Configura restricoes
  glp_add_rows(*lp, nrows);

  // criar uma restricao de capacidade para cada mochila
  row = 1;
  for(k=0;k<I.k;k++){
    sprintf(name,"capacidade_Mochila_%d", row); /* nome das restricoes */
    glp_set_row_name(*lp, row, name);
    glp_set_row_bnds(*lp, row, GLP_UP, 0.0, I.C[k]);
    row++;
  }
  // criar uma restricao de unicidade para cada item
  for(i=0;i<I.n;i++){
    sprintf(name,"unicidade_%d", I.item[i].num); /* nome das restricoes */
    glp_set_row_name(*lp, row, name);
    glp_set_row_bnds(*lp, row, GLP_UP, 0.0, 1.0);
    row++;
  }

  // Configura variaveis
  glp_add_cols(*lp, ncols);

  col = 1;
  for(k=1;k<=I.k;k++){
    for(i=0;i<I.n;i++){
      name[0]='\0';
      sprintf(name,"x%d_%d", i+1, k); /* as variaveis referem-se `as variaveis xi_k para cada item i e cada mochila k */
      glp_set_col_name(*lp, col, name);
      glp_set_col_bnds(*lp, col, GLP_DB, 0.0, 1.0);
      glp_set_obj_coef(*lp, col, I.item[i].valor);
      glp_set_col_kind(*lp, col, GLP_BV); // especifica que a variaval xik eh binaria
      col++;
    }
  }

  // Configura matriz de coeficientes ...
  nz = 1;
  // coeficientes para as restricoes de capacidade
  for(k=1;k<=I.k;k++){ // para cada mochila
    for(i=1;i<=I.n;i++){ // para cada item
      // restr de capacidade
      ia[nz]=k;           // linha (indice da restricao)
      ja[nz]=(k-1)*I.n+i; // coluna (indice da variavel = xik)
      ar[nz]=I.item[i-1].peso; // coeficiente da matriz de coeficientes na linha e coluna
      nz++;
    }
  }

  // Coeficientes para as restricoes de unicidade
	for(k=1;k<=I.n;k++){ // para cada item
		for(i=1;i<=I.k;i++){ // para cada item
			// restr de unicidade
			ia[nz] = I.k+k; 	     // linha (indice da restricao)
			ja[nz]=(i-1)*I.n+k; // coluna (indice da variavel = xik)
			ar[nz] = 1;
			nz++;
		}

	}

  // Carrega PL
  glp_load_matrix(*lp, nz-1, ia, ja, ar);

  // libera memoria
  free(ia); free(ja); free(ar);
  return 1;
}

/* carrega os dados da instancia de entrada */
int carga_instancia(char* filename, Tinstance *I)
{
  FILE *fin;
  int i, capacidade, item, peso;
  double valor;

  fin=fopen(filename,"r");
  if(!fin){
    printf("\nProblema na abertura do arquivo %s\n", filename);
    return 0;
  }

  fscanf(fin, "%d %d", &(I->n), &(I->k));

  // aloca memória
  (*I).C=(int*)malloc(sizeof(int)*((*I).k));
  (*I).item=(Titem*)malloc(sizeof(Titem)*((*I).n));

  for(i=0;i < (*I).k;i++){
    fscanf(fin, "%d", &capacidade);
    (*I).C[i] = capacidade;
  }

  for(i=0;i<(*I).n;i++){
    fscanf(fin, "%d %d %lf", &item, &peso, &valor);
    if(item<1 || item > (*I).n){
      fclose(fin);
      return 0;
    }
    (*I).item[i].num = item;
    (*I).item[i].peso = peso;
    (*I).item[i].valor = valor;
  }

#ifdef DEBUG
  printf("n=%d k=%d\n", (*I).n, (*I).k);
  for(i=0;i< (*I).k;i++){
    printf("C[%d]=%d\n", i+1, (*I).C[i]);
  }
  for(i=0;i< (*I).n;i++){
    printf("p[%d]=%d e v[%d]=%lf\n", (*I).item[i].num, (*I).item[i].peso, (*I).item[i].num, (*I).item[i].valor);
  }
#endif
  fclose(fin);
  return 1;
}

/* libera memoria alocada pelo programa para guardar a instancia */
void  free_instancia(Tinstance I)
{
  free(I.item);
  free(I.C);
}

/* sorteia um numero aleatorio entre [low,high] */
int RandomInteger(int low, int high)
{
    int k;
    double d;

    d = (double)rand() / ((double)RAND_MAX + 1);
    k = d * (high - low + 1);
    return low + k;
}

// callback usada para salvar informações da execução do B&B
void my_callback (glp_tree *tree, void *infop)
{
  int bestnode;
  my_infoT *info;

  info = (my_infoT*) infop;

  switch (glp_ios_reason(tree))
  {
  case GLP_ISELECT:
  case GLP_IBINGO:
    glp_ios_tree_size(tree, &(info->ativos), &(info->nodes), NULL);
    bestnode = glp_ios_best_node(tree);
    info->best_dualBound =  glp_ios_node_bound(tree, bestnode);
    info->best_primalBound = glp_mip_obj_val(info->mip);
    info->gap = glp_ios_mip_gap(tree);
#ifdef DEBUG
    if(info->best_dualBound<1e10)
      printf("DEBUG %d\t%d\t%d\t%.2lf\t%.2lf\t%.2lf\n", info->ativos, info->nodes, bestnode, info->best_primalBound, info->best_dualBound, 100*(info->gap));
    else
      printf("DEBUG %d\t%d\t%d\t%.2lf\t*\t%.2lf\n", info->ativos, info->nodes, bestnode, info->best_primalBound, 100*(info->gap));
#endif
    break;
  default:
    break;
  }
}

/* resolve o problema de PLI usando o GLPK */
double otimiza_PLI(Tinstance I, int tipo, double* x, my_infoT* info)
{
  glp_prob *lp;
  double z, valor;
  glp_smcp param_lp;
  glp_iocp param_ilp;
  int status, i, k;

  // desabilita saidas do GLPK no terminal
  glp_term_out(GLP_OFF);

  // carga do lp
  carga_lp(&lp, I);

  // configura simplex
  glp_init_smcp(&param_lp);
  param_lp.msg_lev = GLP_MSG_ON;

  // configura optimizer
  glp_init_iocp(&param_ilp);
  param_ilp.msg_lev = GLP_MSG_ALL;
  param_ilp.tm_lim = 5000; // tempo limite do solver de PLI
  param_ilp.out_frq = 100;
  // ativa my callback
  param_ilp.cb_func = my_callback;
  param_ilp.cb_info = info;

  // seta mip na estrutura da callback
  info->mip = lp;

  // Executa Solver de PL
  glp_simplex(lp, &param_lp); // resolve o problema relaxado
  if(tipo==2){
    glp_intopt(lp, &param_ilp); // resolve o problema inteiro
  }

  if(tipo==2){
    status=glp_mip_status(lp);
    PRINTF("\nstatus=%d\n", status);
  }
  // Recupera solucao
  if(tipo==1)
    z = glp_get_obj_val(lp);
  else
    z = glp_mip_obj_val(lp);

  for(k=0;k<I.k;k++){
    for(i=0;i<I.n;i++){
      if(tipo==1)
        valor=glp_get_col_prim(lp, k*I.n+i+1); // recupera o valor da variavel xik relaxado (continuo)
      else
        valor=glp_mip_col_val(lp, k*I.n+i+1); // recupera o valor da variavel xik
      if(valor>EPSILON)
        PRINTF("x%d_%d = %.2lf\n", I.item[i].num, k+1, valor);
      x[k*I.n+i]=valor;
    }
  }

#ifdef DEBUG
  // Grava solucao e PL
  PRINTF("\n---LP gravado em mochila.lp e solucao em mochila.sol");
  glp_write_lp(lp, NULL,"mochila.lp");
  if (tipo==1)
    glp_print_sol(lp, "mochila.sol");
  else
    glp_print_mip(lp, "mochila.sol");
#endif
  // Destroi problema
  glp_delete_prob(lp);
  return z;
}

/* heuristicas */
double heuristica(Tinstance I,int tipo, char* entrada, double* x)
{
   	int i, k;
   	bool *vetorMarca;
  	vetorMarca = (bool*)calloc(I.n, sizeof(bool));
   	int count = 0;
   	int r = rand() % (I.n);
   	int *vetorPeso;
   	vetorPeso = (int*)malloc(I.k*sizeof(int));
 	srand(time(NULL));
   	int total = 0;
	int totalMochilas = 0;
	bool *mochilasUsadas;
	mochilasUsadas = (bool*)calloc(I.k, sizeof(bool));
	int *totalItensMochila;
	totalItensMochila = (int*)calloc(I.k, sizeof(int));
	int QuaisItensMochila[I.k][I.n];

	//usado na heuristica 2
	double *ValorporGrama;
	ValorporGrama = (double*)malloc(I.n*sizeof(double));
	int indiceValorporGrama[I.n];
	int indicemenor;
	int j;		
	
	char saida[50];		//saida = entrada.sol
	strcpy(saida, entrada);
	strcat(saida, ".sol");

	FILE *fileptr = fopen(saida, "w");


	for(i = 0; i < I.k; i++){      //vetorpeso = capacidade de kd mochila
		vetorPeso[i] = I.C[i];
 	}

	if(tipo ==3){
		while(count < I.n){
			if(vetorMarca[r] == true){
 				r = rand() % (I.n);
				continue;
			}

			for(i = 0; i< I.k;i++){ //p cada mochila tenta colocar o item_random, comecando pela mochila 1
            			if(I.item[r].peso < vetorPeso[i]){  //item[r] foi levado pela mochila i
                			vetorPeso[i] = vetorPeso[i] - I.item[r].peso;
                			vetorMarca[r] = true;				//marca item usado atualiza total
					total = total + I.item[r].valor;		//guarda qual item foi levado pela mochila i
					mochilasUsadas[i] = true;
					QuaisItensMochila[i][totalItensMochila[i]] = r+1;
					totalItensMochila[i]++;
                			break;
            			}
        		}
			vetorMarca[r] = true;	//marca item se n foi usado
			count++;
			r = rand() % (I.n);
		}

		for(i = 0; i < I.k; i++){
			if(mochilasUsadas[i] == true)	//calcula total de mochilas usadas
			totalMochilas++;	
		}


		//gravar no arquivo
		fprintf(fileptr, "%d %d\n\n", total, totalMochilas);

		for(i = 0; i < totalMochilas; i++){
			fprintf(fileptr, "mochila %d %d\n",i+1,totalItensMochila[i]);
			for(k = 0; k < totalItensMochila[i]; k++){
				fprintf(fileptr, "%d ", QuaisItensMochila[i][k]);
			}
			fprintf(fileptr, "\n");
		}
	}
	else if(tipo == 4){
		
		for (i = 0; i < I.n; i++){
			ValorporGrama[i] = I.item[i].valor/I.item[i].peso ;
		}

		//~selection sort                        
		for(i = 0; i < I.n; i++){        //ordena vetor de valor por grama em ordem decrescente
			indicemenor = 0;
			for(k = 0; k < I.n; k++){
				if(vetorMarca[indicemenor] == true)
					indicemenor++;
				else
					break;
				
			}
			for(j = 1; j < I.n; j++){
				if(ValorporGrama[j] < ValorporGrama[indicemenor] && vetorMarca[j] == false){
					indicemenor = j;
				}
			}
			vetorMarca[indicemenor] = true;
			indiceValorporGrama[I.n-i-1] = indicemenor;
			 
		}//end sort
		

		free(vetorMarca);
		vetorMarca = (bool*)calloc(I.n, sizeof(bool));


		for(i = 0; i < I.k; i++){			//tenta colocar o melhor item na mochila 1 em diante
			for(j = 0; j < I.n; j++){
				if(vetorMarca[indiceValorporGrama[j]] == false && I.item[indiceValorporGrama[j]].peso < vetorPeso[i]){ 
					vetorMarca[indiceValorporGrama[j]] = true;
					vetorPeso[i] = vetorPeso[i] - I.item[indiceValorporGrama[j]].peso;     //entrou na mochila i
					total = total + I.item[indiceValorporGrama[j]].valor;			
					mochilasUsadas[i] = true;						//marca item, att total
					QuaisItensMochila[i][totalItensMochila[i]] = indiceValorporGrama[j]+1;  //guarda item levado mochila i
					totalItensMochila[i]++;
				}
			}
		}

		for(i = 0; i < I.k; i++){	//atualiza total mochilas usadas
			if(mochilasUsadas[i] == true)
				totalMochilas++;	
		}

		//grava no arquivo
		fprintf(fileptr, "%d %d\n\n", total, totalMochilas);

		for(i = 0; i < totalMochilas; i++){
			fprintf(fileptr, "mochila %d %d\n",i+1,totalItensMochila[i]);
			for(k = 0; k < totalItensMochila[i]; k++){
				fprintf(fileptr, "%d ", QuaisItensMochila[i][k]);
			}
			fprintf(fileptr, "\n");
		}
	}

  free(vetorPeso);free(mochilasUsadas);free(totalItensMochila);free(ValorporGrama);free(vetorMarca);
  return total;
}

/* programa principal */
int main(int argc, char **argv)
{
  double z, *x;
  clock_t antes, agora;
  int tipo;
  Tinstance I;
  my_infoT info; // variavel usada pela callback
  FILE* arq_resumido;

  // checa linha de comando
  if(argc<3){
    printf("\nSintaxe: mochila <instancia.txt> <tipo>\n\t<tipo>: 1 = relaxacao linear, 2 = solucao inteira, 3 = heuristica aleatória, 4 = heuristica gulosa\n");
    exit(1);
  }

  // ler a entrada
  if(!carga_instancia(argv[1], &I)){
    printf("\nProblema na carga da instância: %s", argv[1]);
    exit(1);
  }

  tipo = atoi(argv[2]);
  if(tipo<1 || tipo>4){
    printf("Tipo invalido\nUse: tipo = 1 (relaxacao linear), 2 (solucao inteira), 3 (heuristica aleatória), 4 (heuristica gulosa)\n");
    exit(1);
  }

  // aloca memoria para a solucao
  x=(double*)malloc(sizeof(double)*(I.n*I.k));
  antes=clock();
  if(tipo<3){
    z = otimiza_PLI(I, tipo, x, &info);
    char a[100];
    if (tipo == 1)
		sprintf(a, "%s-1-%d.out", argv[1], tipo);
	else
		sprintf(a, "%s-1-0.out", argv[1]);
    arq_resumido = fopen(a, "w+");
  }
  else{
    z = heuristica(I, tipo, argv[1], x);
    char a[100];
	sprintf(a, "%s-2-%d.out", argv[1], tipo - 2);
    arq_resumido = fopen(a, "w+");
  }
  agora=clock();

  fprintf(arq_resumido, "%s;%d;%.0lf;%.3lf;", argv[1], tipo, ((double)agora-antes)/CLOCKS_PER_SEC, z);
  if(tipo == 2)
    fprintf(arq_resumido, "%.0lf", info.best_dualBound);
  if(((double)agora-antes)/CLOCKS_PER_SEC > 4.98)
    fprintf(arq_resumido, ";5\n");
  else fprintf(arq_resumido,    ";10\n");
  // libera memoria alocada
  free_instancia(I);
  free(x);
  return 0;
}

/* eof */
