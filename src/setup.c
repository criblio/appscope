#define _GNU_SOURCE
#include <fcntl.h>
#include <stdlib.h>

#include "loaderop.h"
#include "libdir.h"
#include "setup.h"
#include "scopestdlib.h"

#define BUFSIZE (4096)

#define SCOPE_EXEC_PATH "/usr/lib/appscope" 

#define OPENRC_DIR "/etc/rc.conf"
#define SYSTEMD_DIR "/etc/systemd"
#define INITD_DIR "/etc/init.d"

#define SETUP_SERVICE "LD_PRELOAD=/usr/lib/appscope/libscope.so"

#define SYSTEMD_CFG "[Service]\nEnvironment=LD_PRELOAD=/usr/lib/appscope/libscope.so\n"
#define SYSTEMD_CFG_LEN (sizeof(SYSTEMD_CFG) - 1)

#define INITD_CFG "LD_PRELOAD=/usr/lib/appscope/libscope.so\n"
#define INITD_CFG_LEN (sizeof(INITD_CFG) - 1)

#define OPENRC_CFG "export LD_PRELOAD=/usr/lib/appscope/libscope.so\n"
#define OPENRC_CFG_LEN (sizeof(OPENRC_CFG) - 1)

/*
 * TODO: Refactor this hardcoded path
 * This can be consolidated with libdir.c but required
 * further cleaning like reverse return logic in libdirExists
 */

#define LIBSCOPE_LOC "/usr/lib/appscope/libscope.so"
#define FILTER_LOC "/usr/lib/appscope/scope_filter"
#define PROFILE_SETUP "export LD_PRELOAD=\"/usr/lib/appscope/libscope.so $LD_PRELOAD\"\n"
#define PROFILE_SETUP_LEN (sizeof(PROFILE_SETUP)-1)


typedef enum {
    SERVICE_CFG_ERROR,
    SERVICE_CFG_NEW,
    SERVICE_CFG_EXIST,
} service_cfg_status_t;

struct service_ops {
    bool (*isServiceInstalled)(const char *serviceName);
    service_cfg_status_t (*serviceCfgStatus)(const char *serviceCfgPath);
    service_status_t (*newServiceCfg)(const char *serviceCfgPath);
    service_status_t (*modifyServiceCfg)(const char *serviceCfgPath);
};

/*
 * Check if specified service is installed in Systemd service manager.
 *
 * Returns TRUE if service is installed FALSE otherwise.
 */
static bool
isServiceInstalledSystemD(const char *serviceName) {

    /*
    * List of directories which can contain service configruation file.
    */
    const char *const servicePrefixList[] = {
        "/etc/systemd/system/",
        "/lib/systemd/system/",
        "/run/systemd/system/",
        "/usr/lib/systemd/system/"
    };

    for (int i = 0; i < sizeof(servicePrefixList)/sizeof(char*); ++i) {
        char cfgPath[PATH_MAX] = {0};
        struct stat st = {0};
        if (scope_snprintf(cfgPath, sizeof(cfgPath), "%s/%s.service", servicePrefixList[i], serviceName) < 0) {
            scope_perror("error: isServiceInstalledSystemD, scope_snprintf failed");
            return FALSE;
        }

        if (scope_stat(cfgPath, &st) == 0) {
            return TRUE;
        }
    }
    return FALSE;
}

/*
 * Check if specified service is installed in InitD/OpenRc service manager.
 *
 * Returns TRUE if service is installed FALSE otherwise.
 */
static bool
isServiceInstalledInitDOpenRc(const char *serviceName) {
    char cfgPath[PATH_MAX] = {0};
    struct stat st = {0};
    if (scope_snprintf(cfgPath, sizeof(cfgPath), "/etc/init.d/%s", serviceName) < 0) {
        scope_perror("error: isServiceInstalledInitDOpenRc, scope_snprintf failed");
        return FALSE;
    }

    return (scope_stat(cfgPath, &st) == 0) ? TRUE : FALSE;
}

/*
 * Get Systemd service configuration file status.
 *
 * Returns status of the Service file.
 */
static service_cfg_status_t
serviceCfgStatusSystemD(const char *serviceName) {
    char cfgScript[PATH_MAX] = {0};
    struct stat st = {0};
    service_cfg_status_t ret = SERVICE_CFG_ERROR;

    if (scope_snprintf(cfgScript, sizeof(cfgScript), "/etc/systemd/system/%s.service.d/", serviceName) < 0) {
        scope_perror("error: serviceCfgStatusSystemD, scope_snprintf failed");
        return ret;
    }

    // create service.d directory if it does not exists.
    if (scope_stat(cfgScript, &st) != 0) {
        if (scope_mkdir(cfgScript, 0644) != 0) {
            scope_perror("error: serviceCfgStatusSystemD, scope_mkdir failed");
            return ret;
        }   
    }

    scope_strncat(cfgScript, "env.conf", sizeof("env.conf") - 1);

    if (scope_stat(cfgScript, &st) == 0) {
        ret = SERVICE_CFG_EXIST;
    } else {
        ret = SERVICE_CFG_NEW;
    }

    return ret;
}

/*
 * Get InitD service configuration file status.
 *
 * Returns status of the Service file.
 */
static service_cfg_status_t
serviceCfgStatusInitD(const char *serviceName) {
    char cfgScript[PATH_MAX] = {0};
    struct stat st = {0};
    service_cfg_status_t ret = SERVICE_CFG_ERROR;

    if (scope_snprintf(cfgScript, sizeof(cfgScript), "/etc/sysconfig/%s", serviceName) < 0) {
        scope_perror("error: serviceCfgStatusInitD, scope_snprintf failed");
        return ret;
    }

    if (scope_stat(cfgScript, &st) == 0) {
        ret = SERVICE_CFG_EXIST;
    } else {
        ret = SERVICE_CFG_NEW;
    }
    return ret;
}

/*
 * Get OpenRc service configuration file status.
 *
 * Returns status of the Service file.
 */
static service_cfg_status_t
serviceCfgStatusOpenRc(const char *serviceName) {
    char cfgScript[PATH_MAX] = {0};
    struct stat st = {0};
    service_cfg_status_t ret = SERVICE_CFG_ERROR;

    if (scope_snprintf(cfgScript, sizeof(cfgScript), "/etc/conf.d/%s", serviceName) < 0) {
        scope_perror("error: serviceCfgStatusOpenRc, scope_snprintf failed");
        return ret;
    }

    if (scope_stat(cfgScript, &st) == 0) {
        ret = SERVICE_CFG_EXIST;
    } else {
        ret = SERVICE_CFG_NEW;
    }
    return ret;
}

/*
 * Setup new service configuration for Systemd service.
 *
 * Returns SERVICE_STATUS_SUCCESS if service was setup correctly, other values in case of failure.
 */
static service_status_t
newServiceCfgSystemD(const char *serviceCfgPath) {
    service_status_t res = SERVICE_STATUS_SUCCESS;
    FILE *fPtr = scope_fopen(serviceCfgPath, "a");

    if (fPtr == NULL) {
        scope_perror("error: newServiceCfgSystemD, scope_fopen failed");
        return SERVICE_STATUS_ERROR_OTHER;
    }

    if (scope_fwrite(SYSTEMD_CFG, sizeof(char), SYSTEMD_CFG_LEN, fPtr) < SYSTEMD_CFG_LEN) {
        scope_perror("error: newServiceCfgSystemD, scope_fwrite failed");
        res = SERVICE_STATUS_ERROR_OTHER;
    }

    scope_fclose(fPtr);
    
    return res;
}

/*
 * Setup new service configuration for initD service.
 *
 * Returns SERVICE_STATUS_SUCCESS if service was setup correctly, other values in case of failure.
 */
static service_status_t
newServiceCfgInitD(const char *serviceCfgPath) {
    service_status_t res = SERVICE_STATUS_SUCCESS;
    FILE *fPtr = scope_fopen(serviceCfgPath, "a");

    if (fPtr == NULL) {
        scope_perror("error: newServiceCfgInitD, scope_fopen failed");
        return SERVICE_STATUS_ERROR_OTHER;
    }

    if (scope_fwrite(INITD_CFG, sizeof(char), INITD_CFG_LEN, fPtr) < INITD_CFG_LEN) {
        scope_perror("error: newServiceCfgInitD, scope_fwrite failed");
        res = SERVICE_STATUS_ERROR_OTHER;
    }

    scope_fclose(fPtr);

    return res;
}

/*
 * Setup new service configuration for OpenRc service.
 *
 * Returns SERVICE_STATUS_SUCCESS if service was setup correctly, other values in case of failure.
 */
static service_status_t
newServiceCfgOpenRc(const char *serviceCfgPath) {
    service_status_t res = SERVICE_STATUS_SUCCESS;
    FILE *fPtr = scope_fopen(serviceCfgPath, "a");

    if (fPtr == NULL) {
        scope_perror("error: newServiceCfgOpenRc, scope_fopen failed");
        return SERVICE_STATUS_ERROR_OTHER;
    }

    if (scope_fwrite(OPENRC_CFG, sizeof(char), OPENRC_CFG_LEN, fPtr) < OPENRC_CFG_LEN) {
        scope_perror("error: newServiceCfgOpenRc, scope_fwrite failed");
        res = SERVICE_STATUS_ERROR_OTHER;
    }

    scope_fclose(fPtr);

    return res;
}

/*
 * Check if specified service needs to be configured.
 *
 * Returns TRUE if service is already configured FALSE otherwise.
 */
static bool
isCfgFileConfigured(const char *serviceCfgPath) {
    FILE *fPtr;
    int res = FALSE;
    char buf[BUFSIZE] = {0};

    if ((fPtr = scope_fopen(serviceCfgPath, "r")) == NULL) {
        scope_perror("isCfgFileConfigured scope_fopen failed");
        return res;
    }

    while(scope_fgets(buf, sizeof(buf), fPtr)) {
        if (scope_strstr(buf, SETUP_SERVICE)) {
            res = TRUE;
            break;
        }
    }

    scope_fclose(fPtr);
    return res;
}

/*
 * Modify configuration for Systemd service.
 *
 * Returns SERVICE_STATUS_SUCCESS if service was modified correctly, other values in case of failure.
 */
static service_status_t
modifyServiceCfgSystemd(const char *serviceCfgPath) {
    FILE *readFd;
    FILE *newFd;
    char *tempPath = "/tmp/tmpFile-XXXXXX";
    bool serviceSectionFound = FALSE;

    if ((readFd = scope_fopen(serviceCfgPath, "r")) == NULL) {
        scope_perror("error: modifyServiceCfgSystemd, scope_fopen serviceFile failed");
        return SERVICE_STATUS_ERROR_OTHER;
    }

    if ((newFd = scope_fopen(tempPath, "w+")) == NULL) {
        scope_perror("error: modifyServiceCfgSystemd, scope_fopen tempFile failed");
        scope_fclose(readFd);
        return SERVICE_STATUS_ERROR_OTHER;
    }

    while (!scope_feof(readFd)) {
        char buf[4096] = {0};
        int res = scope_fscanf(readFd, "%s", buf);

        if (scope_strcmp(buf, "[Service]") == 0) {
            serviceSectionFound = TRUE;
            scope_fprintf(newFd, "%s", SYSTEMD_CFG);
        } else if (res == 0){
            scope_fprintf(newFd, "%s ", buf);
        }
    }

    // the file was empty
    if (serviceSectionFound == FALSE) {
        scope_fprintf(newFd, "%s", SYSTEMD_CFG);
    }

    scope_fclose(newFd);
    scope_fclose(readFd);

    if (scope_rename(tempPath, serviceCfgPath)) {
        scope_perror("error: modifyServiceCfgSystemd, scope_rename failed");
    }
    scope_unlink(tempPath);

    return SERVICE_STATUS_SUCCESS;
}

static struct service_ops SystemDService = {
    .isServiceInstalled = isServiceInstalledSystemD,
    .serviceCfgStatus = serviceCfgStatusSystemD,
    .newServiceCfg = newServiceCfgSystemD,
    .modifyServiceCfg= modifyServiceCfgSystemd,
};

static struct service_ops InitDService = {
    .isServiceInstalled = isServiceInstalledInitDOpenRc,
    .serviceCfgStatus = serviceCfgStatusInitD,
    .newServiceCfg = newServiceCfgInitD,
    .modifyServiceCfg= newServiceCfgInitD,
};

static struct service_ops OpenRcService = {
    .isServiceInstalled = isServiceInstalledInitDOpenRc,
    .serviceCfgStatus = serviceCfgStatusOpenRc,
    .newServiceCfg = newServiceCfgOpenRc,
    .modifyServiceCfg= newServiceCfgOpenRc,
};

/*
 * Setup specific service.
 *
 * Returns SERVICE_STATUS_SUCCESS if service was setup correctly, other values in case of failure.
 */
service_status_t
setupService(const char *serviceName) {
    struct stat sb = {0};
    struct service_ops *service;

    char serviceCfgPath[PATH_MAX] = {0};

    service_status_t status;

    if (scope_stat(OPENRC_DIR, &sb) == 0) {
        service = &OpenRcService;
        if (scope_snprintf(serviceCfgPath, sizeof(serviceCfgPath), "/etc/conf.d/%s", serviceName) < 0) {
            scope_perror("error: setupService, scope_snprintf OpenRc failed");
            return SERVICE_STATUS_ERROR_OTHER;
        }
    } else if (scope_stat(SYSTEMD_DIR, &sb) == 0) {
        service = &SystemDService;
        if (scope_snprintf(serviceCfgPath, sizeof(serviceCfgPath), "/etc/systemd/system/%s.service.d/env.conf", serviceName) < 0) {
            scope_perror("error: setupService, scope_snprintf SystemD failed");
            return SERVICE_STATUS_ERROR_OTHER;
        }
    } else if (scope_stat(INITD_DIR, &sb) == 0) {
        service = &InitDService;
        if (scope_snprintf(serviceCfgPath, sizeof(serviceCfgPath), "/etc/sysconfig/%s", serviceName) < 0) {
            scope_perror("error: setupService, scope_snprintf InitD failed");
            return SERVICE_STATUS_ERROR_OTHER;
        }
    } else {
        scope_fprintf(scope_stderr, "error: unknown boot system\n");
        return SERVICE_STATUS_ERROR_OTHER;
    }

    if (service->isServiceInstalled(serviceName) == FALSE) {
        scope_fprintf(scope_stderr, "info: service %s is not installed\n", serviceName);
        return SERVICE_STATUS_NOT_INSTALLED;
    }

    service_cfg_status_t cfgStatus =  service->serviceCfgStatus(serviceName);
    if (cfgStatus == SERVICE_CFG_ERROR) {
        return SERVICE_STATUS_ERROR_OTHER;
    } else if (cfgStatus == SERVICE_CFG_NEW) {
        // Fresh configuration
        status = service->newServiceCfg(serviceCfgPath);
    } else if (isCfgFileConfigured(serviceCfgPath) == FALSE) {
        // Modification of configuration file
        status = service->modifyServiceCfg(serviceCfgPath);
    } else {
        // Service was already setup correctly
        return SERVICE_STATUS_SUCCESS;
    }
    scope_chmod(serviceCfgPath, 0644);
    
    return status;
}


 /*
 * Setup the /etc/profile scope startup script
 * Returns status of operation TRUE in case of success, FALSE otherwise
 */
static bool
setupProfile(void) {
    int fd = scope_open("/etc/profile.d/scope.sh", O_CREAT | O_RDWR | O_TRUNC, 0644);

    if (fd < 0) {
        scope_perror("scope_fopen failed");
        return FALSE;
    }

    if (scope_write(fd, PROFILE_SETUP, PROFILE_SETUP_LEN) != PROFILE_SETUP_LEN) {
        scope_perror("scope_write failed");
        scope_close(fd);
        return FALSE;
    }

    if (scope_close(fd) != 0) {
        scope_perror("scope_fopen failed");
        return FALSE;
    }

    return TRUE;
}

 /*
 * Extract memory to filter file /usr/lib/appscope/scope_filter
 *
 * Returns status of operation TRUE in case of success, FALSE otherwise
 */
static bool
setupExtractFilterFile(void *filterFileMem, size_t filterSize) {
    int filterFd;
    bool status = FALSE;

    if ((filterFd = scope_open(FILTER_LOC, O_RDWR | O_CREAT, 0664)) == -1) {
        scope_perror("scope_open failed");
        return status;
    }

    if (scope_ftruncate(filterFd, filterSize) != 0) {
        goto cleanupDestFd;
    }

    char *dest = scope_mmap(NULL, filterSize, PROT_READ | PROT_WRITE, MAP_SHARED, filterFd, 0);
    if (dest == MAP_FAILED) {
        goto cleanupDestFd;
    }

    scope_memcpy(dest, filterFileMem, filterSize);

    status = TRUE;

cleanupDestFd:

    scope_close(filterFd);

    return status;
}

/*
 * Load File into memory.
 *
 * Returns memory address in case of success, NULL otherwise.
 */
char *
setupLoadFileIntoMem(size_t *size, const char *path)
{
    // Load file into memory
    char *resMem = NULL;
    int fd;

    if (path == NULL) {
        return resMem;
    }

    if ((fd = scope_open(path, O_RDONLY)) == -1) {
        scope_perror("scope_open failed");
        goto closeFd;
    }

    *size = scope_lseek(fd, 0, SEEK_END);
    if (*size == (off_t)-1) {
        scope_perror("scope_lseek failed");
        goto closeFd;
    }

    resMem = scope_mmap(NULL, *size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (resMem == MAP_FAILED) {
        scope_perror("scope_mmap failed");
        resMem = NULL;
        goto closeFd;
    }

closeFd:

    scope_close(fd);

    return resMem;
}

 /*
 * Configure the environment
 * - setup /etc/profile file
 * - extract memory to filter file /usr/lib/appscope/scope_filter
 * - extract libscope.so to /usr/lib/appscope/libscope.so 
 * - patch the library
 * Returns status of operation 0 in case of success, other value otherwise
 */
int
setupConfigure(void *filterFileMem, size_t filterSize) {
    // Setup /etc/profile.d/scope.sh
    if (setupProfile() == FALSE) {
        scope_fprintf(scope_stderr, "setupProfile failed\n");
        return -1;
    }

    // Check for presence of a /usr/lib/appscope directory; add if doesn't exist
    // TODO: not correct, needs to be dynamic
    if (opendir(SCOPE_EXEC_PATH) == NULL) {
        if (mkdir(SCOPE_EXEC_PATH, 0755) == -1) {
            scope_perror("setupConfigure: mkdir failed");
        }
    }

    // Extract the filter file to /usr/lib/appscope/scope_filter
    if (setupExtractFilterFile(filterFileMem, filterSize) == FALSE) {
        scope_fprintf(scope_stderr, "setup filter file failed\n");
        return -1;
    }

    // Extract libscope.so to /usr/lib/appscope/libscope.so
    if (libdirExtractLibraryTo(LIBSCOPE_LOC)) {
        scope_fprintf(scope_stderr, "extract libscope.so failed\n");
        return -1;
    }
    // Patch the library
    if (loaderOpPatchLibrary(LIBSCOPE_LOC) == PATCH_FAILED) {
        scope_fprintf(scope_stderr, "patch libscope.so failed\n");
        return -1;
    }

    return 0;
}
