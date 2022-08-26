#define _GNU_SOURCE
#include <fcntl.h>

#include "service.h"
#include "scopestdlib.h"

#define OPENRC_DIR "/etc/rc.conf"
#define SYSTEMD_DIR "/etc/systemd"
#define INITD_DIR "/etc/init.d"

#define SETUP_SERVICE "LD_PRELOAD=/tmp/libscope.so"

#define SYSTEMD_CFG "[Service]\nEnvironment=LD_PRELOAD=/tmp/libscope.so\n"
#define SYSTEMD_CFG_LEN (sizeof(SYSTEMD_CFG) - 1)

#define INITD_CFG "LD_PRELOAD=/tmp/libscope.so\n"
#define INITD_CFG_LEN (sizeof(INITD_CFG) - 1)

#define OPENRC_CFG "export LD_PRELOAD=/tmp/libscope.so\n"
#define OPENRC_CFG_LEN (sizeof(OPENRC_CFG) - 1)

typedef enum {
    SERVICE_CFG_ERROR,
    SERVICE_CFG_NEW,
    SERVICE_CFG_EXIST,
} service_cfg_status_t;

struct service_ops {
    bool (*isServiceInstalled)(const char* serviceName);
    service_cfg_status_t (*serviceCfgStatus)(const char* serviceCfgPath);
    int (*newServiceCfg)(const char* serviceCfgPath);
    int (*modifyServiceCfg)(const char* serviceCfgPath);
};

/*
 * Check if specified service is installed in Systemd service manager.
 *
 * Returns TRUE if service is installed FALSE otherwise.
 */
static bool
isServiceInstalledSystemD(const char* serviceName) {
    /*
    * List of directories which can contain service configruation file
    *
    */
    const char* const servicePrefixList[] = {
        "/etc/systemd/system/",
        "/lib/systemd/system/",
        "/run/systemd/system/",
        "/usr/lib/systemd/system/"
    };

    for (int i = 0; i < sizeof(servicePrefixList)/sizeof(char*); ++i) {
        char cfgPath[PATH_MAX] = {0};
        struct stat st = {0};
        if (scope_snprintf(cfgPath, sizeof(cfgPath), "%s/%s.service", servicePrefixList[i], serviceName) < 0) {
            scope_perror("scope_snprintf failed");
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
isServiceInstalledInitDOpenRc(const char* serviceName) {
    char cfgPath[PATH_MAX] = {0};
    struct stat st = {0};
    if (scope_snprintf(cfgPath, sizeof(cfgPath), "/etc/init.d/%s", serviceName) < 0) {
        scope_perror("scope_snprintf failed");
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
serviceCfgStatusSystemD(const char* serviceName) {
    char cfgScript[PATH_MAX] = {0};
    struct stat st = {0};
    service_cfg_status_t ret = SERVICE_CFG_ERROR;

    if (scope_snprintf(cfgScript, sizeof(cfgScript), "/etc/systemd/system/%s.service.d/", serviceName) < 0) {
        scope_perror("scope_snprintf failed");
        return ret;
    }

    // create service.d directory if it does not exists 
    if (scope_stat(cfgScript, &st) != 0) {
        if (scope_mkdir(cfgScript, 0644) != 0) {
            scope_perror("scope_mkdir failed");
            return ret;
        }   
    }

    scope_strncat(cfgScript, "env.conf", sizeof("env.conf")-1);

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
serviceCfgStatusInitD(const char* serviceName) {
    char cfgScript[PATH_MAX] = {0};
    struct stat st = {0};
    service_cfg_status_t ret = SERVICE_CFG_ERROR;

    if (scope_snprintf(cfgScript, sizeof(cfgScript), "/etc/sysconfig/%s", serviceName) < 0) {
        scope_perror("scope_snprintf failed");
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
serviceCfgStatusOpenRc(const char* serviceName) {
    char cfgScript[PATH_MAX] = {0};
    struct stat st = {0};
    service_cfg_status_t ret = SERVICE_CFG_ERROR;

    if (scope_snprintf(cfgScript, sizeof(cfgScript), "/etc/conf.d/%s", serviceName) < 0) {
        scope_perror("scope_snprintf failed");
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
 * Returns 0 if service was setup correctly -1 otherwise.
 */
static int
newServiceCfgSystemD(const char* serviceCfgPath) {
    int res = 0;
    FILE *fPtr = scope_fopen(serviceCfgPath, "a");

    if (fPtr == NULL) {
        scope_perror("newServiceCfgSystemD scope_fopen failed");
        return -1;
    }

    if (scope_fwrite(SYSTEMD_CFG, sizeof(char), SYSTEMD_CFG_LEN, fPtr) < SYSTEMD_CFG_LEN) {
        scope_perror("newServiceCfgSystemD scope_fwrite failed");
        res = -1;
    }

    scope_fclose(fPtr);
    
    return res;
}

/*
 * Setup new service configuration for initD service.
 *
 * Returns 0 if service was setup correctly -1 otherwise.
 */
static int
newServiceCfgInitD(const char* serviceCfgPath) {
    int res = 0;
    FILE *fPtr = scope_fopen(serviceCfgPath, "a");

    if (fPtr == NULL) {
        scope_perror("newServiceCfgInitD scope_fopen failed");
        return -1;
    }

    if (scope_fwrite(INITD_CFG, sizeof(char), INITD_CFG_LEN, fPtr) < INITD_CFG_LEN) {
        scope_perror("newServiceCfgInitD scope_fwrite failed");
        res = -1;
    }

    scope_fclose(fPtr);

    return res;
}

/*
 * Setup new service configuration for OpenRc service.
 *
 * Returns 0 if service was setup correctly -1 otherwise.
 */
static int
newServiceCfgOpenRc(const char* serviceCfgPath) {
    int res = 0;
    FILE *fPtr = scope_fopen(serviceCfgPath, "a");

    if (fPtr == NULL) {
        scope_perror("newServiceCfgOpenRc scope_fopen failed");
        return -1;
    }

    if (scope_fwrite(OPENRC_CFG, sizeof(char), OPENRC_CFG_LEN, fPtr) < OPENRC_CFG_LEN) {
        scope_perror("newServiceCfgOpenRc scope_fwrite failed");
        res = -1;
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
isCfgFileConfigured(const char* serviceCfgPath) {
    FILE *fPtr;
    int res = FALSE;
    char buf[4096] = {0};

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
 * Returns 0 if service was modified correctly -1 otherwise.
 */
static int
modifyServiceCfgSystemd(const char* serviceCfgPath) {
    FILE *readFd;
    FILE *newFd;
    char * tempPath = "/tmp/tmpFile-XXXXXX";
    bool serviceSectionFound = FALSE;

    if ((readFd = scope_fopen(serviceCfgPath, "r")) == NULL) {
        scope_perror("scope_fopen failed");
        return -1;
    }

    if ((newFd = scope_fopen(tempPath, "w+")) == NULL) {
        scope_perror("scope_fopen failed");
        scope_fclose(readFd);
        return -1;
    }

    while (!scope_feof(readFd)) {
        char buf[4096] = {0};
        scope_fscanf(readFd, "%s", buf);

        if (scope_strcmp(buf, "[Service]") == 0) {
            serviceSectionFound = TRUE;
            scope_fprintf(newFd, "%s", SYSTEMD_CFG);
        } else {
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
        scope_perror("scope_rename failed");
    }
    scope_unlink(tempPath);

    return -1;
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
 * Setup specific service
 *
 * Returns 0 if service was setup correctly -1 otherwise.
 */
int
serviceSetup(const char* serviceName) {
    struct stat sb = {0};
    struct service_ops *service;

    char serviceCfgPath[PATH_MAX] = {0};

    int status;

    if (scope_stat(OPENRC_DIR, &sb) == 0) {
        service = &OpenRcService;
        if (scope_snprintf(serviceCfgPath, sizeof(serviceCfgPath), "/etc/conf.d/%s", serviceName) < 0) {
            scope_perror("scope_snprintf failed");
            return -1;
        }
    } else if (scope_stat(SYSTEMD_DIR, &sb) == 0) {
        service = &SystemDService;
        if (scope_snprintf(serviceCfgPath, sizeof(serviceCfgPath), "/etc/systemd/system/%s.service.d/env.conf", serviceName) < 0) {
            scope_perror("scope_snprintf failed");
            return -1;
        }
    } else if (scope_stat(INITD_DIR, &sb) == 0) {
        service = &InitDService;
        if (scope_snprintf(serviceCfgPath, sizeof(serviceCfgPath), "/etc/sysconfig/%s", serviceName) < 0) {
            scope_perror("scope_snprintf failed");
            return -1;
        }
    } else {
        scope_fprintf(scope_stderr, "error: unknown boot system\n");
        return -1;
    }

    if (service->isServiceInstalled(serviceName) == FALSE) {
        scope_fprintf(scope_stderr, "error: service %s is not installed\n", serviceName);
        return -1;
    }

    service_cfg_status_t cfgStatus =  service->serviceCfgStatus(serviceName);
    if (cfgStatus == SERVICE_CFG_ERROR) {
        return -1;
    } else if (cfgStatus == SERVICE_CFG_NEW) {
        // Fresh configuration
        status = service->newServiceCfg(serviceCfgPath);
        scope_chmod(serviceCfgPath, 0644);
    } else if (isCfgFileConfigured(serviceCfgPath) == FALSE) {
        // Modification of configuration file
        status = service->modifyServiceCfg(serviceCfgPath);
        scope_chmod(serviceCfgPath, 0644);
    } else {
        // Service was already setup correctly
        status = 0;
    }
    
    return status;
}
