#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <pmmintrin.h>


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

void productoCuaterniones( float *operando1, float *operando2, float
    *destino );

void productoSumaCuaterniones( float *operando1, float *operando2, float
    *destino );


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
    start_counter();

    // Se almacena en el vector 'c' la multiplicación de los vectores 'a' y 'b'
    for( i = 0; i < n; i++ )
    {
        productoCuaterniones( a + i * 4, b + i * 4, c + i * 4 );
    }

    // Se inicializan los valores del cuaternión 'dp' a '0'
    dp[ 0 ] = 0;
    dp[ 1 ] = 0;
    dp[ 2 ] = 0;
    dp[ 3 ] = 0;

    // Se realiza sobre el cuaternión 'dp' la suma de la multiplicación de cada
    // cuaternión del vector 'c' por sí mismo
    for( i = 0; i < n; i++ )
    {
        productoSumaCuaterniones( c + i * 4, c + i * 4, dp );
    }

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


void productoCuaterniones( float *operando1, float *operando2, float
    *destino )
{
    destino[ 0 ] = operando1[ 0 ] * operando2[ 0 ] - operando1[ 1 ] *
        operando2[ 1 ] - operando1[ 2 ] * operando2[ 2 ] - operando1[ 3 ] *
        operando2[ 3 ];

    destino[ 1 ] = operando1[ 0 ] * operando2[ 1 ] + operando1[ 1 ] *
        operando2[ 0 ] + operando1[ 2 ] * operando2[ 3 ] - operando1[ 3 ] *
        operando2[ 2 ];

    destino[ 2 ] = operando1[ 0 ] * operando2[ 2 ] - operando1[ 1 ] *
        operando2[ 3 ] + operando1[ 2 ] * operando2[ 0 ] + operando1[ 3 ] *
        operando2[ 1 ];

    destino[ 3 ] = operando1[ 0 ] * operando2[ 3 ] + operando1[ 1 ] *
        operando2[ 2 ] - operando1[ 2 ] * operando2[ 1 ] + operando1[ 3 ] *
        operando2[ 0 ];
}


void productoSumaCuaterniones( float *operando1, float *operando2, float
    *destino )
{
    destino[ 0 ] += operando1[ 0 ] * operando2[ 0 ] - operando1[ 1 ] *
        operando2[ 1 ] - operando1[ 2 ] * operando2[ 2 ] - operando1[ 3 ] *
        operando2[ 3 ];

    destino[ 1 ] += operando1[ 0 ] * operando2[ 1 ] + operando1[ 1 ] *
        operando2[ 0 ] + operando1[ 2 ] * operando2[ 3 ] - operando1[ 3 ] *
        operando2[ 2 ];

    destino[ 2 ] += operando1[ 0 ] * operando2[ 2 ] - operando1[ 1 ] *
        operando2[ 3 ] + operando1[ 2 ] * operando2[ 0 ] + operando1[ 3 ] *
        operando2[ 1 ];

    destino[ 3 ] += operando1[ 0 ] * operando2[ 3 ] + operando1[ 1 ] *
        operando2[ 2 ] - operando1[ 2 ] * operando2[ 1 ] + operando1[ 3 ] *
        operando2[ 0 ];
}
