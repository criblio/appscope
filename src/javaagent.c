#include "dbg.h"
#include "os.h"
#include "state.h"
#include <jni.h>
#include <jvmti.h>
#include "javabci.h"

static jmethodID g_mid_Object_hashCode          = NULL;
static jmethodID g_mid_SSLSocketImpl_getSession = NULL;
static jmethodID g_mid_AppOutputStream___write  = NULL;
static jmethodID g_mid_AppInputStream___read    = NULL;
static jfieldID  g_fid_AppOutputStream_socket   = NULL;
static jfieldID  g_fid_AppInputStream_socket    = NULL;
static jmethodID g_mid_ByteBuffer_array         = NULL;
static jmethodID g_mid_ByteBuffer_position      = NULL;
static jmethodID g_mid_ByteBuffer_limit         = NULL;
static jmethodID g_mid_SSLEngineImpl___wrap     = NULL;
static jmethodID g_mid_SSLEngineImpl___unwrap   = NULL;
static jmethodID g_mid_SSLEngineImpl_getSession = NULL;

static void logJvmtiError(jvmtiEnv *jvmti, jvmtiError errnum, const char *str) {
    char buf[1024];
    char *errnum_str = NULL;
    (*jvmti)->GetErrorName(jvmti, errnum, &errnum_str);
    snprintf(buf, sizeof(buf), "ERROR: JVMTI: [%d] %s - %s\n", errnum, (errnum_str == NULL ? "Unknown": errnum_str), (str == NULL? "" : str));
    scopeLog(buf, -1, CFG_LOG_ERROR);
}

static void 
clearJniException(JNIEnv *jni)
{
    jboolean flag = (*jni)->ExceptionCheck(jni);
    if (flag) (*jni)->ExceptionClear(jni);
}

static void 
initJniGlobals(JNIEnv *jni) 
{
    if (g_mid_Object_hashCode != NULL) return;
    jclass objectClass             = (*jni)->FindClass(jni, "java/lang/Object");
    g_mid_Object_hashCode          = (*jni)->GetMethodID(jni, objectClass, "hashCode", "()I");

    jclass sslSocketImplClass      = (*jni)->FindClass(jni, "sun/security/ssl/SSLSocketImpl");
    g_mid_SSLSocketImpl_getSession = (*jni)->GetMethodID(jni, sslSocketImplClass, "getSession", "()Ljavax/net/ssl/SSLSession;");

    jclass byteBufferClass         = (*jni)->FindClass(jni, "java/nio/ByteBuffer");
    jclass bufferClass             = (*jni)->FindClass(jni, "java/nio/Buffer");
    g_mid_ByteBuffer_array         = (*jni)->GetMethodID(jni, byteBufferClass, "array", "()[B");
    g_mid_ByteBuffer_position      = (*jni)->GetMethodID(jni, bufferClass, "position", "()I");
    g_mid_ByteBuffer_limit         = (*jni)->GetMethodID(jni, bufferClass, "limit", "()I");
}

static void 
initAppOutputStreamGlobals(JNIEnv *jni)
{
    if (g_mid_AppOutputStream___write != NULL) return;
    jclass appOutputStreamClass    = (*jni)->FindClass(jni, "sun/security/ssl/AppOutputStream");
    if (appOutputStreamClass == NULL) {
        // JDK 11
        appOutputStreamClass = (*jni)->FindClass(jni, "sun/security/ssl/SSLSocketImpl$AppOutputStream");
    }
    g_mid_AppOutputStream___write  = (*jni)->GetMethodID(jni, appOutputStreamClass, "__write", "([BII)V");
    g_fid_AppOutputStream_socket   = (*jni)->GetFieldID(jni, appOutputStreamClass, "c", "Lsun/security/ssl/SSLSocketImpl;");
    if (g_fid_AppOutputStream_socket == NULL) {
        //support for JDK 9, JDK 10
        g_fid_AppOutputStream_socket = (*jni)->GetFieldID(jni, appOutputStreamClass, "socket", "Lsun/security/ssl/SSLSocketImpl;");
    }
    if (g_fid_AppOutputStream_socket == NULL) {
        //support for JDK 11 - 14
        g_fid_AppOutputStream_socket = (*jni)->GetFieldID(jni, appOutputStreamClass, "this$0", "Lsun/security/ssl/SSLSocketImpl;");
    }
    if (g_fid_AppOutputStream_socket == NULL) {
        scopeLog("unable to find an SSLSocket field in AppOutputStream class", -1, CFG_LOG_DEBUG);
    }
    clearJniException(jni);
}

static void
initAppInputStreamGlobals(JNIEnv *jni)
{
    if (g_mid_AppInputStream___read != NULL) return;
    jclass appInputStreamClass     = (*jni)->FindClass(jni, "sun/security/ssl/AppInputStream");
    if (appInputStreamClass == NULL) {
        // JDK 11
        appInputStreamClass  = (*jni)->FindClass(jni, "sun/security/ssl/SSLSocketImpl$AppInputStream");
    }
    g_mid_AppInputStream___read    = (*jni)->GetMethodID(jni, appInputStreamClass, "__read", "([BII)I");
    g_fid_AppInputStream_socket    = (*jni)->GetFieldID(jni, appInputStreamClass, "c", "Lsun/security/ssl/SSLSocketImpl;");
    if (g_fid_AppInputStream_socket == NULL) {
        //support for JDK 9, JDK 10
        g_fid_AppInputStream_socket = (*jni)->GetFieldID(jni, appInputStreamClass, "socket", "Lsun/security/ssl/SSLSocketImpl;");
    }
    if (g_fid_AppInputStream_socket == NULL) {
        //support for JDK 11 - 14
        g_fid_AppInputStream_socket = (*jni)->GetFieldID(jni, appInputStreamClass, "this$0", "Lsun/security/ssl/SSLSocketImpl;");
    }
    if (g_fid_AppInputStream_socket == NULL) {
        scopeLog("unable to find an SSLSocket field in AppInputStream class", -1, CFG_LOG_DEBUG);
    }
    clearJniException(jni);
}

static void
initSSLEngineImplGlobals(JNIEnv *jni) 
{
    if (g_mid_SSLEngineImpl___unwrap != NULL) return;
    jclass sslEngineImplClass      = (*jni)->FindClass(jni, "sun/security/ssl/SSLEngineImpl");
    g_mid_SSLEngineImpl___unwrap   = (*jni)->GetMethodID(jni, sslEngineImplClass, "__unwrap", "(Ljava/nio/ByteBuffer;[Ljava/nio/ByteBuffer;II)Ljavax/net/ssl/SSLEngineResult;");
    g_mid_SSLEngineImpl___wrap     = (*jni)->GetMethodID(jni, sslEngineImplClass, "__wrap", "([Ljava/nio/ByteBuffer;IILjava/nio/ByteBuffer;)Ljavax/net/ssl/SSLEngineResult;");
    g_mid_SSLEngineImpl_getSession = (*jni)->GetMethodID(jni, sslEngineImplClass, "getSession", "()Ljavax/net/ssl/SSLSession;");
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
    if (name != NULL && 
        (strcmp(name, "sun/security/ssl/AppOutputStream") == 0 || 
         strcmp(name, "sun/security/ssl/SSLSocketImpl$AppOutputStream") == 0)) {

        scopeLog("installing Java SSL hooks for AppOutputStream class...", -1, CFG_LOG_DEBUG);

        java_class_t *classInfo = javaReadClass(class_data);

        int methodIndex = javaFindMethodIndex(classInfo, "write", "([BII)V");
        if (methodIndex == -1) {
            javaDestroy(&classInfo);
            scopeLog("ERROR: 'write' method not found in AppOutputStream class\n", -1, CFG_LOG_ERROR);
            return;
        }
        javaCopyMethod(classInfo, classInfo->methods[methodIndex], "__write");
        javaConvertMethodToNative(classInfo, methodIndex);

        unsigned char *dest;
        (*jvmti_env)->Allocate(jvmti_env, classInfo->length, &dest);
        javaWriteClass(dest, classInfo);

        *new_class_data_len = classInfo->length;
        *new_class_data = dest;
        javaDestroy(&classInfo);
    }

    if (name != NULL && 
        (strcmp(name, "sun/security/ssl/AppInputStream") == 0 ||
         strcmp(name, "sun/security/ssl/SSLSocketImpl$AppInputStream") == 0)) {

        scopeLog("installing Java SSL hooks for AppInputStream class...", -1, CFG_LOG_DEBUG);

        java_class_t *classInfo = javaReadClass(class_data);

        int methodIndex = javaFindMethodIndex(classInfo, "read", "([BII)I");
        if (methodIndex == -1) {
            javaDestroy(&classInfo);
            scopeLog("ERROR: 'read' method not found in AppInputStream class\n", -1, CFG_LOG_ERROR);
            return;
        }
        javaCopyMethod(classInfo, classInfo->methods[methodIndex], "__read");
        javaConvertMethodToNative(classInfo, methodIndex);

        unsigned char *dest;
        (*jvmti_env)->Allocate(jvmti_env, classInfo->length, &dest);
        javaWriteClass(dest, classInfo);

        *new_class_data_len = classInfo->length;
        *new_class_data = dest;
        javaDestroy(&classInfo);
    }

    if (name != NULL && strcmp(name, "sun/security/ssl/SSLEngineImpl") == 0) {
        scopeLog("installing Java SSL hooks for SSLEngineImpl class...", -1, CFG_LOG_DEBUG);

        java_class_t *classInfo = javaReadClass(class_data);

        int methodIndex = javaFindMethodIndex(classInfo, "wrap", "([Ljava/nio/ByteBuffer;IILjava/nio/ByteBuffer;)Ljavax/net/ssl/SSLEngineResult;");
        if (methodIndex == -1) {
            javaDestroy(&classInfo);
            scopeLog("ERROR: 'wrap' method not found in SSLEngineImpl class\n", -1, CFG_LOG_ERROR);
            return;
        }
        javaCopyMethod(classInfo, classInfo->methods[methodIndex], "__wrap");
        javaConvertMethodToNative(classInfo, methodIndex);

        methodIndex = javaFindMethodIndex(classInfo, "unwrap", "(Ljava/nio/ByteBuffer;[Ljava/nio/ByteBuffer;II)Ljavax/net/ssl/SSLEngineResult;");
         if (methodIndex == -1) {
            javaDestroy(&classInfo);
            scopeLog("ERROR: 'unwrap' method not found in SSLEngineImpl class\n", -1, CFG_LOG_ERROR);
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
doJavaProtocol(JNIEnv *jni, jobject session, jbyteArray buf, jint offset, jint len, metric_t src)
{
    jint  hash      = (*jni)->CallIntMethod(jni, session, g_mid_Object_hashCode);
    jbyte *byteBuf  = (*jni)->GetPrimitiveArrayCritical(jni, buf, 0);
    doProtocol((uint64_t)hash, -1, &byteBuf[offset], (size_t)(len - offset), src, BUF);
    (*jni)->ReleasePrimitiveArrayCritical(jni, buf, buf, 0);
}

JNIEXPORT jobject JNICALL 
Java_sun_security_ssl_SSLEngineImpl_unwrap(JNIEnv *jni, jobject obj, jobject netData, jobjectArray appData, jint offset, jint len)
{
    initJniGlobals(jni);
    initSSLEngineImplGlobals(jni);

    //call the original method
    jobject res = (*jni)->CallObjectMethod(jni, obj, g_mid_SSLEngineImpl___unwrap, netData, appData, offset, len);

    jobject session = (*jni)->CallObjectMethod(jni, obj, g_mid_SSLEngineImpl_getSession);
    for(int i=offset;i<len - offset;i++) {
        jobject bufEl  = (*jni)->GetObjectArrayElement(jni, appData, i);
        jint pos       = (*jni)->CallIntMethod(jni, bufEl, g_mid_ByteBuffer_position);
        jbyteArray buf = (*jni)->CallObjectMethod(jni, bufEl, g_mid_ByteBuffer_array);
        doJavaProtocol(jni, session, buf, 0, pos, TLSRX);
    }
    return res;
}

JNIEXPORT jobject JNICALL 
Java_sun_security_ssl_SSLEngineImpl_wrap(JNIEnv *jni, jobject obj, jobjectArray appData, jint offset, jint len, jobject netData) 
{
    initJniGlobals(jni);
    initSSLEngineImplGlobals(jni);

    jobject session = (*jni)->CallObjectMethod(jni, obj, g_mid_SSLEngineImpl_getSession);
    for(int i=offset;i<len - offset;i++) {
        jobject bufEl  = (*jni)->GetObjectArrayElement(jni, appData, i);
        jint pos       = (*jni)->CallIntMethod(jni, bufEl, g_mid_ByteBuffer_position);
        jint limit     = (*jni)->CallIntMethod(jni, bufEl, g_mid_ByteBuffer_limit);
        jbyteArray buf = (*jni)->CallObjectMethod(jni, bufEl, g_mid_ByteBuffer_array);
        doJavaProtocol(jni, session, buf, pos, limit, TLSTX);
    }
    
    //call the original method
    jobject res = (*jni)->CallObjectMethod(jni, obj, g_mid_SSLEngineImpl___wrap, appData, offset, len, netData);
    return res;
} 



JNIEXPORT void JNICALL 
Java_sun_security_ssl_AppOutputStream_write(JNIEnv *jni, jobject obj, jbyteArray buf, jint offset, jint len) 
{
    initJniGlobals(jni);
    initAppOutputStreamGlobals(jni);

    jobject session;
    if (g_fid_AppOutputStream_socket != NULL) {
        jobject socket  = (*jni)->GetObjectField(jni, obj, g_fid_AppOutputStream_socket);
        session = (*jni)->CallObjectMethod(jni, socket, g_mid_SSLSocketImpl_getSession);
    } else {
        session = obj;
    }
    
    doJavaProtocol(jni, session, buf, offset, len, TLSTX);
    
    //call the original method
    (*jni)->CallVoidMethod(jni, obj, g_mid_AppOutputStream___write, buf, offset, len);
}


JNIEXPORT void JNICALL 
Java_sun_security_ssl_SSLSocketImpl_00024AppOutputStream_write(JNIEnv *jni, jobject obj, jbyteArray buf, jint offset, jint len) {
    Java_sun_security_ssl_AppOutputStream_write(jni, obj, buf, offset, len);
}


JNIEXPORT jint JNICALL 
Java_sun_security_ssl_AppInputStream_read(JNIEnv *jni, jobject obj, jbyteArray buf, jint offset, jint len) 
{
    initJniGlobals(jni);
    initAppInputStreamGlobals(jni);

    //call the original method
    jint res = (*jni)->CallIntMethod(jni, obj, g_mid_AppInputStream___read, buf, offset, len);

    jobject session = NULL;

    if (g_fid_AppInputStream_socket != NULL) {
        jobject socket  = (*jni)->GetObjectField(jni, obj, g_fid_AppInputStream_socket);
        session = (*jni)->CallObjectMethod(jni, socket, g_mid_SSLSocketImpl_getSession);
    } else {
        session = obj;
    }
    
    doJavaProtocol(jni, session, buf, offset, res, TLSRX);

    return res;
}

JNIEXPORT void JNICALL 
Java_sun_security_ssl_SSLSocketImpl_00024AppInputStream_read(JNIEnv *jni, jobject obj, jbyteArray buf, jint offset, jint len) {
    Java_sun_security_ssl_AppInputStream_read(jni, obj, buf, offset, len);
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