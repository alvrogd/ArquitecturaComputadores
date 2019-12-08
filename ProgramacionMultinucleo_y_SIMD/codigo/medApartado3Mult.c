#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <pmmintrin.h>


/* Características de la CPU */

// Múltiplo del que tienen que ser las direcciones de memoria a reservar para
// un alineado correcto para SIMD
#define ALIN_MULT 16

/* Macros varias */
#define FALSE 0
#define TRUE 1


/* Prototipos de las funciones a emplear */
void start_counter();
double get_counter();
double mhz();

void inicializarVectorCuaternion( float **vector, size_t numElementos, int
    valoresAleatorios );
void liberarVectorCuaternion( float **vector );


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

    // Número de cuaterniones contenidos en cada array
    size_t n;

    // Orden de magnitud del tamaño de los vectores de input
    size_t q;

    // Vectores de cuaterniones de valores aleatorios (input)
    float *a;
    float *b;

    // Vector auxiliar de cuaterniones
    float *c;

    // Cuaternión sobre el que realizar la computación (output)
    float dp[ 4 ];

    // Variable sobre la que contabilizar el tiempo transcurrido
    double ck;

    // Contadores
    int i;

    // Variable auxiliar para indicar el cuaternión "a" actual
    __m128 qA;

    // Variable auxiliar para indicar el cuaternión "b" actual
    __m128 qB;

    // Variable auxiliar para almacenar el resultado del producto de los
    // cuaterniones qA * qB
    __m128 qC;

    // Cuaternión final sobre el que se realiza el cómputo
    __m128 qDP;

    // Cuaterniones auxiliares para la realización del producto
    __m128 qA0;
    __m128 qA1;
    __m128 qA2;
    __m128 qA3;

    // Cuaternión a 0
    __m128 q0;

    // Cuaternión auxiliar para guardar valores intermedios de una suma
    __m128 qSUM;

    // Cuaternión auxiliar para guardar valores intermedios de la multiplación
    __m128 qMULT;


    /***** Argumentos *****/

    if( argc < 2 )
    {
        printf( "Número de valores incorrecto. Uso: %s <q>\n",
            argv[ 0 ] );
        exit( EXIT_FAILURE );
    }

    // Se obtiene el valor de 'q' dado
    q = atoi( argv[ 1 ] );

    if( q <= 0 )
    {
        printf( "El valor de q debe ser mayor que 0\n" );
        exit( EXIT_FAILURE );
    }


    /***** Inicialización *****/

    // Se obtiene una semilla para la generación de números aleatorios
    srand( ( unsigned )time( NULL ) );

    // Se calcula el tamaño final de los vectores de input
    n = ( int )pow( 10, q );

    // Se inicializan los vectores de cuaterniones
    inicializarVectorCuaternion( &a, n, TRUE );
    inicializarVectorCuaternion( &b, n, TRUE );
    inicializarVectorCuaternion( &c, n, FALSE );


    /***** Computación *****/

    // Se inicia el medidor de tiempo
    ck = 0;
    qDP = _mm_setzero_ps();
    q0 = _mm_setzero_ps();
    start_counter();

    // Se almacena en el vector 'c' la multiplicación de los vectores 'a' y 'b'
    for( i = 0; i < n; i++ )
    {
        qA = _mm_load_ps(a + i * 4);
        qB = _mm_load_ps(b + i * 4);

        qA0 = _mm_shuffle_ps(qA, qA, _MM_SHUFFLE(0,0,0,0));
        qA1 = _mm_shuffle_ps(qA, qA, _MM_SHUFFLE(1,1,1,1));
        qA2 = _mm_shuffle_ps(qA, qA, _MM_SHUFFLE(2,2,2,2));
        qA3 = _mm_shuffle_ps(qA, qA, _MM_SHUFFLE(3,3,3,3));

        qC = _mm_mul_ps(qA0, qB);
        qC = _mm_addsub_ps(qC, _mm_mul_ps(qA1, _mm_shuffle_ps(qB, qB,
                                        _MM_SHUFFLE(2, 3, 0, 1))));

        // Se realiza el intercambio de los elementos con signo más y menos para
        // poder realizar el addsub
        qC = _mm_shuffle_ps(qC, qC, _MM_SHUFFLE(2, 3, 1, 0));

        qC = _mm_addsub_ps(qC, _mm_mul_ps(qA2, _mm_shuffle_ps(qB, qB,
                                        _MM_SHUFFLE(0, 1, 3, 2))));

        qC = _mm_shuffle_ps(qC, qC, _MM_SHUFFLE(2, 1, 3, 0));
        qC = _mm_addsub_ps(qC, _mm_mul_ps(qA3, _mm_shuffle_ps(qB, qB,
                                        _MM_SHUFFLE(0, 2, 1, 3))));

        // Recolocación del cuaternión qC
        qC = _mm_shuffle_ps(qC, qC, _MM_SHUFFLE(3, 1, 2, 0));
    }

    // Se inicializan los valores del cuaternión 'dp' a '0'

    // Se realiza sobre el cuaternión 'dp' la suma de la multiplicación de cada
    // cuaternión del vector 'c' por sí mismo
    for( i = 0; i < n; i++ )
    {
        // Se guardan las componentes del cuaternión iterado
        qA0 = _mm_shuffle_ps(qC, qC, _MM_SHUFFLE(0, 0, 0, 0));

        // Se realiza la suma de a + a
        qSUM = _mm_add_ps(qC, qC);

        // Se obtiene el cuaternion (2a0*a0, 2a0*a1, 2a0*a2, 2a0*a3)
        qMULT = _mm_mul_ps(qSUM, qA0);

        // Se obtiene el productor de a * a
        qA1 = _mm_mul_ps(qC, qC);

        // Se obtiene (a0a0+a1a1, a2a2+a3a3, a0a0+a1a1, a2a2+a3a3)
        qSUM = _mm_hadd_ps(qA1, qA1);

        // Se obtiene (a0a0+a1a1, a2a2+a3a3, 0, 0)
        qSUM = _mm_shuffle_ps(qSUM, q0, _MM_SHUFFLE(3,2,1,0));

        // Se obtiene (a0a0+a1a1, 0, 0, 0)
        qA1 = _mm_shuffle_ps(qSUM, q0, _MM_SHUFFLE(0,0,2,0));

        // Se realiza la resta sobre qMULT
        qMULT = _mm_sub_ps(qMULT, qA1);

        // Se obtiene (a2a2+a3a3, 0, 0, 0)
        qA1 = _mm_shuffle_ps(qSUM, q0, _MM_SHUFFLE(0,0,2,1));

        // Se resta
        qMULT = _mm_sub_ps(qMULT, qA1);

        // Se le añade al cuaternion suma
        qDP = _mm_add_ps(qDP, qMULT);
    }

    _mm_store_ps(dp, qDP);

    // Se finaliza el medidor de tiempo
    ck = get_counter();

    printf( "%d,%lu,%1.10lf\n", atoi( argv[ 2 ] ), q, ck );

    printf( "Resultado: [%f, %f, %f, %f]\n", dp[ 0 ], dp[ 1 ], dp[ 2 ],
        dp[ 3 ] );

    // Se libera la memoria reservada
    liberarVectorCuaternion( &a );
    liberarVectorCuaternion( &b );
    liberarVectorCuaternion( &c );


    return( EXIT_SUCCESS );
}


void inicializarVectorCuaternion( float **vector, size_t numElementos, int
    valoresAleatorios )
{
    // Contador
    int i;


    // Se reserva la memoria necesaria para el vector de cuaterniones
    if( ( *vector = _mm_malloc( numElementos * 4 * sizeof( float ),
        ALIN_MULT ) ) == NULL )
    {
        perror( "Reserva de memoria del vector de cuaterniones fallida" );
        exit( EXIT_FAILURE );
    }

    if( valoresAleatorios == TRUE )
    {
        // Y se genera en cada posición de los cuaterniones de los vectores un
        // valor entre 1 y 2
        for( i = 0; i < numElementos; i++ )
        {
            *( *vector + i * 4 ) = ( ( double )rand() / RAND_MAX + 1 ) *
                pow( -1, rand() % 2 );
            *( *vector + i * 4 + 1 ) = ( ( double )rand() / RAND_MAX + 1 ) *
                pow( -1, rand() % 2 );
            *( *vector + i * 4 + 2 ) = ( ( double )rand() / RAND_MAX + 1 ) *
                pow( -1, rand() % 2 );
            *( *vector + i * 4 + 3 ) = ( ( double )rand() / RAND_MAX + 1 ) *
                pow( -1, rand() % 2 );
        }
    }

    else
    {
        // En caso contrario, se inicializan los valores a '0'
        for( i = 0; i < numElementos; i++ )
        {
            *( *vector + i * 4 ) = 0;
            *( *vector + i * 4 + 1 ) = 0;
            *( *vector + i * 4 + 2 ) = 0;
            *( *vector + i * 4 + 3 ) = 0;
        }
    }
}


void liberarVectorCuaternion( float **vector )
{
    _mm_free( *vector );
}
