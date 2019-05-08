.ONESHELL:
SHELL=/bin/bash

define PROJECT_HELP_MSG
Makefile for testing notebooks
Make sure you have edited the dev_env_template files and renamed it to .dev_env
All the variables loaded in this makefile must come from the .dev_env file

Usage:
	make test					run all notebooks
	make clean					delete env and remove files
endef
export PROJECT_HELP_MSG
env_location=./../../not_shared/general_master.env
PWD:=$(shell pwd)
include ${env_location}


help:
	echo "$$PROJECT_HELP_MSG" | less

setup:
ifndef TENANT_ID
	@echo starting interactive login
	az login  --output none
	az account set --subscription ${SUBSCRIPTION_ID}
	az account list --all --refresh -o table
else
	@echo using service principal login
	az login -t ${TENANT_ID} --service-principal -u ${SP_USERNAME} --password ${SP_PASSWORD}
endif

test-notebook000:
	crt_notebook=000_RegularR_RealTime_Scripts_and_SDK_setup.ipynb
	@echo Testing $${crt_notebook}
	papermill $${crt_notebook} test.ipynb \
		--log-output \
		--no-progress-bar \
		--kernel python3 \
		-p subscription_id ${SUBSCRIPTION_ID} \
		-p resource_group ${RESOURCE_GROUP} \
		-p workspace_name ${WORKSPACE_NAME} \
		-p compute_context_vm_user_name ${COMPUTE_CONTEXT_VM_USER_NAME} \
		-p compute_context_vm_fqdn ${COMPUTE_CONTEXT_VM_FQDN} \
		-p compute_context_vm_ssh_port ${COMPUTE_CONTEXT_VM_SSH_PORT} \
		-p compute_context_vm_pwd ${COMPUTE_CONTEXT_VM_PWD} \
# 		-p sp_tenant_id ${SPTID} \
# 		-p sp_application_id ${SPAPPID} \
# 		-p sp_password ${SPPWD} \

test-notebook010:
	crt_notebook=010_RegularR_RealTime_test_score_in_docker.ipynb
	@echo Testing $${crt_notebook}
	papermill $${crt_notebook} test.ipynb \
		--log-output \
		--no-progress-bar \
		--kernel python3 


test-notebook020:
	crt_notebook=020_RegularR_RealTime_deploy_ACI_AKS.ipynb
	@echo Testing $${crt_notebook}
	papermill $${crt_notebook} test.ipynb \
		--log-output \
		--no-progress-bar \
		--kernel python3 \
		-p version_of_image_to_deploy 5
    
