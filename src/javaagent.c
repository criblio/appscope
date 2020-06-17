#include "dbg.h"
#include "os.h"
#include "state.h"
#include <jni.h>
#include <jvmti.h>
#include "javabci.h"

static jmethodID g_mid_Object_hashCode          = NULL;
static jmethodID g_mid_SSLSocketImpl_getSession = NULL;
static jmethodID g_mid_AppOutputStream___write  = NULL;
static jfieldID  g_fid_AppOutputStream_socket   = NULL;
static jmethodID g_mid_AppInputStream___read    = NULL;
static jfieldID  g_fid_AppInputStream_socket    = NULL;

static void logJvmtiError(jvmtiEnv *jvmti, jvmtiError errnum, const char *str) {
    char buf[1024];
    char *errnum_str = NULL;
    (*jvmti)->GetErrorName(jvmti, errnum, &errnum_str);
    snprintf(buf, sizeof(buf), "ERROR: JVMTI: [%d] %s - %s\n", errnum, (errnum_str == NULL ? "Unknown": errnum_str), (str == NULL? "" : str));
    scopeLog(buf, -1, CFG_LOG_ERROR);
}

static void printBuf(JNIEnv* jni, char *method, jobject buf_param,  int offset, int len) {
    printf("METHOD = %s, LEN = %d. OFFSET = %d \nBUFFER=", method, len, offset);

    jbyte *buf = (*jni)->GetPrimitiveArrayCritical(jni, buf_param, 0);
    for(int i=0; i<len; i++) {
        if (buf[i + offset]==0) 
            printf("0");
        else 
            printf("%c",  buf[i + offset]);
    }
    printf("\n\n");
    (*jni)->ReleasePrimitiveArrayCritical(jni, buf_param, buf, 0);
}

void JNICALL 
ClassFileLoadHook(jvmtiEnv *jvmti_env,
    JNIEnv* jni,
    jclass class_being_redefined,
    jobject loader,
    const char* name,
    jobject protection_domain,
    jint class_data_len,
    const unsigned char* class_data,
    jint* new_class_data_len,
    unsigned char** new_class_data) 
{
    // if (name != NULL && strcmp(name, "sun/security/ssl/AppOutputStream") == 0) {
    //     scopeLog("installing Java SSL hooks for AppOutputStream class...", -1, CFG_LOG_DEBUG);

    //     jclass objectClass = (*jni)->FindClass(jni, "java/lang/Object");
    //     g_mid_Object_hashCode = (*jni)->GetMethodID(jni, objectClass, "hashCode", "()I");

    //     jclass sslSocketImplClass = (*jni)->FindClass(jni, "sun/security/ssl/SSLSocketImpl");
    //     g_mid_SSLSocketImpl_getSession = (*jni)->GetMethodID(jni, sslSocketImplClass, "getSession", "()Ljavax/net/ssl/SSLSession;");
    
    //     java_class_t *classInfo = javaReadClass(class_data);

    //     int methodIndex = javaFindMethodIndex(classInfo, "write", "([BII)V");
    //     javaCopyMethod(classInfo, classInfo->methods[methodIndex], "__write");
    //     javaConvertMethodToNative(classInfo, methodIndex);

    //     unsigned char *dest;
    //     (*jvmti_env)->Allocate(jvmti_env, classInfo->length, &dest);
    //     javaWriteClass(dest, classInfo);

    //     *new_class_data_len = classInfo->length;
    //     *new_class_data = dest;
    //     javaDestroy(&classInfo);
    // }

    // if (name != NULL && strcmp(name, "sun/security/ssl/AppInputStream") == 0) {
    //     scopeLog("installing Java SSL hooks for AppInputStream class...", -1, CFG_LOG_DEBUG);

    //     java_class_t *classInfo = javaReadClass(class_data);

    //     int methodIndex = javaFindMethodIndex(classInfo, "read", "([BII)I");
    //     javaCopyMethod(classInfo, classInfo->methods[methodIndex], "__read");
    //     javaConvertMethodToNative(classInfo, methodIndex);

    //     unsigned char *dest;
    //     (*jvmti_env)->Allocate(jvmti_env, classInfo->length, &dest);
    //     javaWriteClass(dest, classInfo);

    //     *new_class_data_len = classInfo->length;
    //     *new_class_data = dest;
    //     javaDestroy(&classInfo);
    // }

    if (name != NULL && strcmp(name, "sun/security/ssl/SSLEngineImpl") == 0) {
        printf("installing Java SSL hooks for SSLEngineImpl class...\n");
        scopeLog("installing Java SSL hooks for SSLEngineImpl class...", -1, CFG_LOG_DEBUG);

        java_class_t *classInfo = javaReadClass(class_data);

        int methodIndex = javaFindMethodIndex(classInfo, "wrap", "([Ljava/nio/ByteBuffer;IILjava/nio/ByteBuffer;)Ljavax/net/ssl/SSLEngineResult;");
        if (methodIndex == -1) {
            javaDestroy(&classInfo);
            printf("method wrap not found\n");
            return;
        }
        javaCopyMethod(classInfo, classInfo->methods[methodIndex], "__wrap");
        javaConvertMethodToNative(classInfo, methodIndex);

        methodIndex = javaFindMethodIndex(classInfo, "unwrap", "(Ljava/nio/ByteBuffer;[Ljava/nio/ByteBuffer;II)Ljavax/net/ssl/SSLEngineResult;");
         if (methodIndex == -1) {
            javaDestroy(&classInfo);
            printf("method unwrap not found\n");
            return;
        }
        javaCopyMethod(classInfo, classInfo->methods[methodIndex], "__unwrap");
        javaConvertMethodToNative(classInfo, methodIndex);
        

        unsigned char *dest;
        (*jvmti_env)->Allocate(jvmti_env, classInfo->length, &dest);
        javaWriteClass(dest, classInfo);

        *new_class_data_len = classInfo->length;
        *new_class_data = dest;
        javaDestroy(&classInfo);
    }

}

static void 
doJavaProtocol(JNIEnv *jni, jobject socket, jbyteArray buf, jint offset, jint len, metric_t src)
{
    jobject session   = (*jni)->CallObjectMethod(jni, socket, g_mid_SSLSocketImpl_getSession);
    jint    hash      = (*jni)->CallIntMethod(jni, session, g_mid_Object_hashCode);
    jbyte   *byteBuf  = (*jni)->GetPrimitiveArrayCritical(jni, buf, 0);
    doProtocol((uint64_t)hash, -1, &byteBuf[offset], (size_t)(len - offset), src, BUF);
    (*jni)->ReleasePrimitiveArrayCritical(jni, buf, buf, 0);
}

JNIEXPORT jobject JNICALL 
Java_sun_security_ssl_SSLEngineImpl_unwrap(JNIEnv *jni, jobject obj, jobject netData, jobjectArray appData, jint offset, jint len)
{
    jclass appOutputStreamClass   = (*jni)->FindClass(jni, "sun/security/ssl/SSLEngineImpl");
    jmethodID wrapMid = (*jni)->GetMethodID(jni, appOutputStreamClass, "__unwrap", "(Ljava/nio/ByteBuffer;[Ljava/nio/ByteBuffer;II)Ljavax/net/ssl/SSLEngineResult;");

    //call the original method
    jobject res = (*jni)->CallObjectMethod(jni, obj, wrapMid, netData, appData, offset, len);

    jclass byteBufferClass   = (*jni)->FindClass(jni, "java/nio/ByteBuffer");
    jmethodID arrayMid = (*jni)->GetMethodID(jni, byteBufferClass, "array", "()[B");

    jclass bufferClass   = (*jni)->FindClass(jni, "java/nio/Buffer");
    jmethodID positionMid = (*jni)->GetMethodID(jni, bufferClass, "position", "()I");

    for(int i=0;i<len;i++) {
        jobject bufEl = (*jni)->GetObjectArrayElement(jni, appData, i);

        jint pos = (*jni)->CallIntMethod(jni, bufEl, positionMid);
        jbyteArray buf = (*jni)->CallObjectMethod(jni, bufEl, arrayMid);

        jbyte   *byteBuf  = (*jni)->GetPrimitiveArrayCritical(jni, buf, 0);
        doProtocol((uint64_t)1111, -1, byteBuf, (size_t)pos, TLSRX, BUF);
        (*jni)->ReleasePrimitiveArrayCritical(jni, buf, buf, 0);
        
        printBuf(jni, "read", buf, 0, pos);
    }
    
    return res;
}

JNIEXPORT jobject JNICALL 
Java_sun_security_ssl_SSLEngineImpl_wrap(JNIEnv *jni, jobject obj, jobjectArray appData, jint offset, jint len, jobject netData) 
{
    jclass appOutputStreamClass   = (*jni)->FindClass(jni, "sun/security/ssl/SSLEngineImpl");
    jmethodID wrapMid = (*jni)->GetMethodID(jni, appOutputStreamClass, "__wrap", "([Ljava/nio/ByteBuffer;IILjava/nio/ByteBuffer;)Ljavax/net/ssl/SSLEngineResult;");

    jclass byteBufferClass   = (*jni)->FindClass(jni, "java/nio/ByteBuffer");
    jmethodID arrayMid = (*jni)->GetMethodID(jni, byteBufferClass, "array", "()[B");

    jclass bufferClass   = (*jni)->FindClass(jni, "java/nio/Buffer");
    jmethodID positionMid = (*jni)->GetMethodID(jni, bufferClass, "position", "()I");
    jmethodID limitMid = (*jni)->GetMethodID(jni, bufferClass, "limit", "()I");
        
    for(int i=0;i<len;i++) {
        jobject bufEl = (*jni)->GetObjectArrayElement(jni, appData, i);

        jint pos = (*jni)->CallIntMethod(jni, bufEl, positionMid);
        jint limit = (*jni)->CallIntMethod(jni, bufEl, limitMid);
        jbyteArray buf = (*jni)->CallObjectMethod(jni, bufEl, arrayMid);

        jbyte   *byteBuf  = (*jni)->GetPrimitiveArrayCritical(jni, buf, 0);
        doProtocol((uint64_t)1111, -1, &byteBuf[pos], (size_t)(limit), TLSTX, BUF);
        (*jni)->ReleasePrimitiveArrayCritical(jni, buf, buf, 0);
        
        //printBuf(jni, "write", buf, pos, limit);
    }
    
    //call the original method
    jobject res = (*jni)->CallObjectMethod(jni, obj, wrapMid, appData, offset, len, netData);
    return res;
} 

JNIEXPORT void JNICALL 
Java_sun_security_ssl_AppOutputStream_write(JNIEnv *jni, jobject obj, jbyteArray buf, jint offset, jint len) 
{
    if (g_mid_AppOutputStream___write == NULL) {
        jclass appOutputStreamClass   = (*jni)->FindClass(jni, "sun/security/ssl/AppOutputStream");
        g_mid_AppOutputStream___write = (*jni)->GetMethodID(jni, appOutputStreamClass, "__write", "([BII)V");
        g_fid_AppOutputStream_socket  = (*jni)->GetFieldID(jni, appOutputStreamClass, "c", "Lsun/security/ssl/SSLSocketImpl;");
    }

    jobject socket  = (*jni)->GetObjectField(jni, obj, g_fid_AppOutputStream_socket);
    doJavaProtocol(jni, socket, buf, offset, len, TLSTX);
    printBuf(jni, "write", buf, offset, len);
    
    //call the original method
    (*jni)->CallVoidMethod(jni, obj, g_mid_AppOutputStream___write, buf, offset, len);
} 

JNIEXPORT jint JNICALL 
Java_sun_security_ssl_AppInputStream_read(JNIEnv *jni, jobject obj, jbyteArray buf, jint offset, jint len) 
{
    if (g_mid_AppInputStream___read == NULL) {
        jclass appInputStreamClass   = (*jni)->FindClass(jni, "sun/security/ssl/AppInputStream");
        g_mid_AppInputStream___read  = (*jni)->GetMethodID(jni, appInputStreamClass, "__read", "([BII)I");
        g_fid_AppInputStream_socket  = (*jni)->GetFieldID(jni, appInputStreamClass, "c", "Lsun/security/ssl/SSLSocketImpl;");
    }

    //call the original method
    jint res = (*jni)->CallIntMethod(jni, obj, g_mid_AppInputStream___read, buf, offset, len);

    jobject socket = (*jni)->GetObjectField(jni, obj, g_fid_AppInputStream_socket);
    doJavaProtocol(jni, socket, buf, offset, res, TLSRX);

    printBuf(jni, "read", buf, offset, res);

    return res;
}

JNIEXPORT jint JNICALL 
Agent_OnLoad(JavaVM *jvm, char *options, void *reserved) 
{
    jvmtiError error;
    jvmtiEnv *env;

    scopeLog("Initializing Java agent", -1, CFG_LOG_INFO);

    jint result = (*jvm)->GetEnv(jvm, (void **) &env, JVMTI_VERSION_1_0);
    if (result != 0) {
        scopeLog("ERROR: GetEnv failed\n", -1, CFG_LOG_ERROR);
        return JNI_ERR;
    }

    jvmtiCapabilities capabilities;
    memset(&capabilities,0, sizeof(capabilities));

    capabilities.can_generate_all_class_hook_events = 1;
    error = (*env)->AddCapabilities(env, &capabilities);
    if (error != JVMTI_ERROR_NONE) {
        logJvmtiError(env, error, "AddCapabilities");
        return JNI_ERR;
    }

    error = (*env)->SetEventNotificationMode(env, JVMTI_ENABLE, JVMTI_EVENT_CLASS_FILE_LOAD_HOOK, NULL);
    if (error != JVMTI_ERROR_NONE) {
        logJvmtiError(env, error, "SetEventNotificationMode");
        return JNI_ERR;
    }
   
    jvmtiEventCallbacks callbacks;
    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.ClassFileLoadHook = &ClassFileLoadHook;
    error = (*env)->SetEventCallbacks(env, &callbacks, sizeof(callbacks));
    if (error != JVMTI_ERROR_NONE) {
        logJvmtiError(env, error, "SetEventCallbacks");
        return JNI_ERR;
    }

    return JNI_OK;
}

void
initJavaAgent() {
    //TODO: 
    // - check if we are in a java process
    // - preserve existing _JAVA_OPTIONS
    char *var = getenv("LD_PRELOAD");
    if (var != NULL) {
        char buf[1024];
        snprintf(buf, sizeof(buf), "-agentpath:%s", var);
        int result = setenv("_JAVA_OPTIONS", buf, 1);
        if (result) {
            scopeLog("ERROR: Could not set _JAVA_OPTIONS failed\n", -1, CFG_LOG_ERROR);
        }
    }
}