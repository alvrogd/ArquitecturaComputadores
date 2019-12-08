#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pmmintrin.h>
#include <time.h>
#include <unistd.h>


/*
Info caché de un core:

  - L1 de datos:
    - Tam: 32 K
    - Vías: 8
    - Tam línea: 64
    - Num conjuntos: 64
    - Tam conjunto: 512 bytes

  - L2 unificada:
    - Tam: 256 K
    - Vías: 4
    - Tam línea: 64
    - Num conjuntos: 1024
    - Tam conjunto: 256 bytes

  - L3 unificada:
    - Tam: 3072 K
    - Vías: 12
    - Tam línea: 64
    - Num conjuntos: 4096
    - Tam conjunto: 768 bytes
*/


/* Características de la CPU */
#define S1 64 * 8
#define S2 1024 * 4
#define CLS 64


/* Macros varias */
#define NUM_S 10
#define DOUBLES_LINEA CLS / sizeof( double )


/* Declaración de las funciones a emplear */
void start_counter();
double get_counter();
double mhz();


/* Initialize the cycle counter */
static unsigned cyc_hi = 0;
static unsigned cyc_lo = 0;


 /* Set *hi and *lo to the high and low order bits of the cycle counter.
 Implementation requires assembly code to use the rdtsc instruction. */
 void access_counter(unsigned *hi, unsigned *lo)
 {
   asm("rdtsc; movl %%edx,%0; movl %%eax,%1" /* Read cycle counter */
     : "=r" (*hi), "=r" (*lo) /* and move results to */
     : /* No input */ /* the two outputs */
     : "%edx", "%eax");
 }


 /* Record the current value of the cycle counter. */
 void start_counter()
 {
   access_counter(&cyc_hi, &cyc_lo);
 }


 /* Return the number of cycles since the last call to start_counter. */
 double get_counter()
 {
   unsigned ncyc_hi, ncyc_lo;
   unsigned hi, lo, borrow;
   double result;

   /* Get cycle counter */
   access_counter(&ncyc_hi, &ncyc_lo);

   /* Do double precision subtraction */
   lo = ncyc_lo - cyc_lo;
   borrow = lo > ncyc_lo;
   hi = ncyc_hi - cyc_hi - borrow;
   result = (double) hi * (1 << 30) * 4 + lo;

   if (result < 0) {
     fprintf(stderr, "Error: counter returns neg value: %.0f\n", result);
   }

   return result;
 }


double mhz(int verbose, int sleeptime)
{
  double rate;

  start_counter();
  sleep(sleeptime);
  rate = get_counter() / (1e6*sleeptime);
  if (verbose)
  printf("\n Processor clock rate = %.1f MHz\n", rate);

  return rate;
}


/* Main */
int main(int argc, char **argv)
{
  /* Variables a emplear */

  double ck;

  // Valor D
  int D;

  // Valores L
  int valoresL[] = { S1 / 2, 3 * S1 / 2, S2 / 2, 3 * S2 / 4, 2 * S2, 4 *
    S2, 8 * S2 };

  // Valor R
  int R;

  // Valores e
  int *e;

  // Valores S
  double valoresS[ NUM_S ];

  // Valores A
  double *valoresA;

  // TC calculado
  int TC;

  // Suma temporal
  double suma;

  // Contadores
  int i;
  int j;

  // Fichero en el que guardar el resultado
  FILE *fichero;


  /***** Argumentos *****/

  if( argc < 3 )
  {
    printf( "Número de valores incorrecto. Uso: %s <D> <L>",
      argv[ 0 ] );
    exit( EXIT_FAILURE );
  }


  /***** Inicialización *****/

  // Se obtiene una semilla para la generación de números aleatorios
  srand( ( unsigned )time( NULL ) );

  // Se obtiene el valor de D directamente de los argumentos
  D = atoi( argv[ 1 ] );

  // Se obtiene el valor para R; si D supera el número de doubles por línea,
  // es necesario limitarlo a dicho valor

  // Hay siete medidas para cada valor de D, para lo cual es necesario
  // considerar el L iterado para la reserva de memoria del vector A.
  // Se multiplica el número de líneas a leer por la cantidad de doubles
  // por línea, y se divide entre el paso D para saber con cuántos valores
  // se efectuará la reducción de suma de punto flotante
  if( D <= DOUBLES_LINEA )
  {
      R = ( int )ceil( valoresL[ atoi( argv[ 2 ] ) ] * DOUBLES_LINEA / D );
  }
  else
  {
      R = valoresL[ atoi( argv[ 2 ] ) ];
  }

  // Se obtienen los valores para e, reservando en primer lugar espacio
  // para las posiciones pertinentes
  if( ( e = ( int * )malloc( R * sizeof( int ) ) ) == NULL )
  {
      perror( "Reserva de memoria fallida" );
      exit( EXIT_FAILURE );
  }

  // El límite superior es el número de valores con los que efectuar la
  // suma
  for( i = 0; i < R; i++ )
  {
    // Se consideran el número de posición y el paso dado
    e[ i ] = i * D;
  }

  // Se obtienen valores para A; se debe reservar memoria también para los
  // elementos intermedios, multiplicando por ello el número de operandos
  // de la suma por el paso que se ha tenido en cuenta
  /*if( D <= DOUBLES_LINEA )
  {*/
        TC = ( R - 1 ) * D + 1;
  /*}
  else {
      // Si D es mayor que el número de doubles por línea, debe emplarse otra
      // fórmula puesto que no es posible reservar tan sólo L líneas continuas
      // por el paso tan grande
      TC = 1 + ( valoresL[ atoi( argv[ 2 ] ) - 1 ) * D;
  }*/

  // Se alinea la reserva al inicio de una línea de la caché
  if( ( valoresA = _mm_malloc( TC * sizeof( double ), CLS ) ) == NULL )
  {
      perror( "Reserva de memoria fallida" );
      exit( EXIT_FAILURE );
  }

  // Y se genera en cada posición un valor entre 1 y 2
  for( i = 0; i < TC; i++ )
  {
      valoresA[ i ] = ( ( double )rand() / RAND_MAX + 1 ) *
        pow( -1, rand() % 2 );

      //printf( "valor A: %f \t[%d %d %d]\n", valoresA[ i ], D, valoresL[atoi(argv[2])], i );
  }


  /***** Pruebas *****/

  // Se registra el contador de la CPU
  start_counter();

  // Se emplean los índices calculados
  for( i = 0, suma = 0; i < R; i++ )
  {
      // Se realiza el acceso a memoria
      suma += valoresA[ e[ i ] ];
  }

  // Se almacena la reducción de punto flotante
  valoresS[ 0 ] = suma;

  // Se registran los ciclos transcurridos desde el registro del contador
  ck = get_counter();

  //printf("\n Clocks=%1.10lf \n",ck);

  /* Esta rutina imprime a frecuencia de reloxo estimada coas rutinas
  start_counter/get_counter */
  //mhz(1,1);

  // Se abre el archivo
  if( ( fichero = fopen( "resultado.csv", "a" ) ) == NULL )
  {
      perror( "No se ha podido abrir el fichero para escritura" );
      exit( EXIT_FAILURE );
  }

  // Se imprimen el valor de L, el número de ciclos medios por acceso y el
  // valor de D en un formato csv que vaya a interpretar el graficador
  fprintf( fichero, "%d,%1.10lf,%d\n", valoresL[ atoi( argv[ 2 ] ) ],
    ck / ( NUM_S * R ), D );

  // Se imprimen las sumas, porque podría darse el caso de que el compilador
  // decida optimizar el programa si nunca se acceden a los datos
  for( i = 0; i < NUM_S; i++ )
  {
      printf( "%f\n", valoresS[ i ] );
  }


  return( EXIT_SUCCESS );
}
