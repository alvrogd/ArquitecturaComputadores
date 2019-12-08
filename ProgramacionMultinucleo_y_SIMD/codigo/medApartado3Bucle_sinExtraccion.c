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


/* Estructura en la que almacenar un vector de cuaterniones */
struct VectorCuaterniones
{
    float *w;
    float *x;
    float *y;
    float *z;
};


/* Prototipos de las funciones a emplear */
void start_counter();
double get_counter();
double mhz();

void inicializarVectorCuaternion( struct VectorCuaterniones *vector, size_t
    numElementos, int valoresAleatorios );
void liberarVectorCuaternion( struct VectorCuaterniones *vector );


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
    struct VectorCuaterniones a;
    struct VectorCuaterniones b;

    // Vector auxiliar de cuaterniones
    struct VectorCuaterniones c;

    // Cuaternión sobre el que realizar la computación (output)
    float dp[ 4 ];

    // Cuaterniones sobre los que almacenar temporalmente los resultados del
    // segundo bucle en lugar de efecutar constantemente reducciones
    __m128 dp0, dp1, dp2, dp3;

    // Variable sobre la que contabilizar el tiempo transcurrido
    double ck;

    // Variables auxiliares en las que almacenar elementos de cuaterniones
    __m128 a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3;

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

    // Se almacena en el vector 'c' la multiplicación de los vectores 'a' y
    // 'b'; en cada iteración del bucle se computan 4 multiplicaciones de
    // cuaterniones
    for( i = 0; i < n; i += 4 )
    {
        // Se guardan las componentes de los cuatro cuaterniones iterados en
        // cada vector
        a0 = _mm_load_ps( a.w + i );
        a1 = _mm_load_ps( a.x + i );
        a2 = _mm_load_ps( a.y + i );
        a3 = _mm_load_ps( a.z + i );

        b0 = _mm_load_ps( b.w + i );
        b1 = _mm_load_ps( b.x + i );
        b2 = _mm_load_ps( b.y + i );
        b3 = _mm_load_ps( b.z + i );

        // Se realiza el producto de los cuatro primeros cuaterniones por los
        // cuatro del vector 'b'
        c0 = _mm_sub_ps( _mm_sub_ps( _mm_sub_ps( _mm_mul_ps( a0, b0 ), _mm_mul_ps( a1, b1 ) ), _mm_mul_ps( a2, b2 ) ), _mm_mul_ps( a3, b3 ) );
        c1 = _mm_sub_ps( _mm_add_ps( _mm_add_ps( _mm_mul_ps( a0, b1 ), _mm_mul_ps( a1, b0 ) ), _mm_mul_ps( a2, b3 ) ), _mm_mul_ps( a3, b2 ) );
        c2 = _mm_add_ps( _mm_add_ps( _mm_sub_ps( _mm_mul_ps( a0, b2 ), _mm_mul_ps( a1, b3 ) ), _mm_mul_ps( a2, b0 ) ), _mm_mul_ps( a3, b1 ) );
        c3 = _mm_add_ps( _mm_sub_ps( _mm_add_ps( _mm_mul_ps( a0, b3 ), _mm_mul_ps( a1, b2 ) ), _mm_mul_ps( a2, b1 ) ), _mm_mul_ps( a3, b0 ) );

        /* Referencia

        *( c + i * 4 ) = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3;
        *( c + i * 4 + 1 ) = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2;
        *( c + i * 4 + 2 ) = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1;
        *( c + i * 4 + 3 ) = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0; */

        // Se extraen los valores de los cuatro cuaterniones resultado
        // componente a componente y se almacenan en el vector 'c'

        // Componente 'w'
        _mm_store_ps( c.w + i, c0 );

        // Componente 'x'
        _mm_store_ps( c.x + i, c1 );

        // Componente 'y'
        _mm_store_ps( c.y + i, c2 );

        // Componente 'z'
        _mm_store_ps( c.z + i, c3 );
    }

    // Se inicializan a (0, 0, 0, 0) los cuaterniones auxiliares para este
    // bucle
    dp0 = _mm_setzero_ps();
    dp1 = _mm_setzero_ps();
    dp2 = _mm_setzero_ps();
    dp3 = _mm_setzero_ps();

    // Se realiza sobre el cuaternión 'dp' la suma de la multiplicación de cada
    // cuaternión del vector 'c' por sí mismo
    for( i = 0; i < n; i += 4 )
    {
        // Se guardan las componentes de los cuatro cuaterniones iterados en el
        // vector 'c'
        a0 = _mm_load_ps( c.w + i );
        a1 = _mm_load_ps( c.x + i );
        a2 = _mm_load_ps( c.y + i );
        a3 = _mm_load_ps( c.z + i );

        // Se realiza el producto de los cuatro primeros cuaterniones por sí
        // mismos
        c0 = _mm_sub_ps( _mm_sub_ps( _mm_sub_ps( _mm_mul_ps( a0, a0 ), _mm_mul_ps( a1, a1 ) ), _mm_mul_ps( a2, a2 ) ), _mm_mul_ps( a3, a3 ) );
        c1 = _mm_mul_ps( _mm_add_ps( a0, a0 ), a1 );
        c2 = _mm_mul_ps( _mm_add_ps( a0, a0 ), a2 );
        c3 = _mm_mul_ps( _mm_add_ps( a0, a0 ), a3 );

        /* Referencia

        dp[ 0 ] += c0 * c0 - c1 * c1 - c2 * c2 - c3 * c3;
        dp[ 1 ] += ( c0 + c0 ) * c1;
        dp[ 2 ] += ( c0 + c0 ) * c2;
        dp[ 3 ] += ( c0 + c0 ) * c3; */

        // Se añaden los 4 valores calculados para cada componente i a la
        // variable auxiliar del componente i
        dp0 = _mm_add_ps( dp0, c0 );
        dp1 = _mm_add_ps( dp1, c1 );
        dp2 = _mm_add_ps( dp2, c2 );
        dp3 = _mm_add_ps( dp3, c3 );
    }

    // Se extraen los valores de los cuaterniones auxiliares componente a
    // componente y se añaden a las componentes del cuaternión 'dp'

    // Componente 'w'
    dp[ 0 ] = _mm_cvtss_f32( _mm_hadd_ps( _mm_hadd_ps( dp0, dp0 ), dp0 ) );

    // Componente 'x'
    dp[ 1 ] = _mm_cvtss_f32( _mm_hadd_ps( _mm_hadd_ps( dp1, dp1 ), dp1 ) );

    // Componente 'y'
    dp[ 2 ] = _mm_cvtss_f32( _mm_hadd_ps( _mm_hadd_ps( dp2, dp2 ), dp2 ) );

    // Componente 'z'
    dp[ 3 ] = _mm_cvtss_f32( _mm_hadd_ps( _mm_hadd_ps( dp3, dp3 ), dp3 ) );

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


void inicializarVectorCuaternion( struct VectorCuaterniones *vector, size_t
    numElementos, int valoresAleatorios )
{
    // Contador
    int i;


    // Se reserva la memoria necesaria para el vector de cuaterniones
    if( ( vector->w = _mm_malloc( numElementos * sizeof( float ), ALIN_MULT ) )
        == NULL )
    {
        perror( "Reserva de memoria del vector de cuaterniones (componente "
                "'w') fallida" );
        exit( EXIT_FAILURE );
    }

    if( ( vector->x = _mm_malloc( numElementos * sizeof( float ), ALIN_MULT ) )
        == NULL )
    {
        perror( "Reserva de memoria del vector de cuaterniones (componente "
                "'x') fallida" );
        exit( EXIT_FAILURE );
    }

    if( ( vector->y = _mm_malloc( numElementos * sizeof( float ), ALIN_MULT ) )
        == NULL )
    {
        perror( "Reserva de memoria del vector de cuaterniones (componente "
                "'y') fallida" );
        exit( EXIT_FAILURE );
    }

    if( ( vector->z = _mm_malloc( numElementos * sizeof( float ), ALIN_MULT ) )
        == NULL )
    {
        perror( "Reserva de memoria del vector de cuaterniones (componente "
                "'z') fallida" );
        exit( EXIT_FAILURE );
    }

    if( valoresAleatorios == TRUE )
    {
        // Y se genera en cada posición de los cuaterniones de los vectores un
        // valor entre 1 y 2
        for( i = 0; i < numElementos; i++ )
        {
            *( vector->w + i ) = ( ( double )rand() / RAND_MAX + 1 ) *
                pow( -1, rand() % 2 );
            *( vector->x + i ) = ( ( double )rand() / RAND_MAX + 1 ) *
                pow( -1, rand() % 2 );
            *( vector->y + i ) = ( ( double )rand() / RAND_MAX + 1 ) *
                pow( -1, rand() % 2 );
            *( vector->z + i ) = ( ( double )rand() / RAND_MAX + 1 ) *
                pow( -1, rand() % 2 );
        }
    }

    else
    {
        // En caso contrario, se inicializan los valores a '0'
        for( i = 0; i < numElementos; i++ )
        {
            *( vector->w + i ) = 0;
            *( vector->x + i ) = 0;
            *( vector->y + i ) = 0;
            *( vector->z + i ) = 0;
        }
    }
}


void liberarVectorCuaternion( struct VectorCuaterniones *vector )
{
    _mm_free( vector->w );
    _mm_free( vector->x );
    _mm_free( vector->y );
    _mm_free( vector->z );
}
