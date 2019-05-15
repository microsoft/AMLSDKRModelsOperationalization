.ONESHELL:
SHELL=/bin/bash

define PROJECT_HELP_MSG
Makefile for testing notebooks
Make sure you have edited the dev_env_template files and renamed it to /not_shared/general_master.env
All the variables loaded in this makefile must come from the /not_shared/general_master.env

Usage:
	make test-notebook000					runs first notebook
	make clean					delete env and remove files
endef
export PROJECT_HELP_MSG
PWD:=$(shell pwd)
env_location:=./../../not_shared/general_master.env
include ${env_location}
aml_sdk_conda_env_name:=aml_sdk_conda_env 
image_name:=aml-sdk_docker_imageazcliazdevops

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
	conda env create --name $(aml_sdk_conda_env_name) -f docker/aml_sdk_conda_dep_file.yml


test-notebook000:
	source activate $(aml_sdk_conda_env_name)
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
	source activate $(aml_sdk_conda_env_name)
	crt_notebook=010_RegularR_RealTime_test_score_in_docker.ipynb
	@echo Testing $${crt_notebook}
	papermill $${crt_notebook} test.ipynb \
		--log-output \
		--no-progress-bar \
		--kernel python3 


test-notebook020:
	source activate $(aml_sdk_conda_env_name)
	crt_notebook=020_RegularR_RealTime_deploy_ACI_AKS.ipynb
	@echo Testing $${crt_notebook}
	papermill $${crt_notebook} test.ipynb \
		--log-output \
		--no-progress-bar \
		--kernel python3 \
		-p version_of_image_to_deploy 1

build:
	echo Will build $(image_name)    
	docker build -t $(image_name) -f docker/dockerfile_aml-sdk  docker 
    
jupyter:
	build
	docker run \
		--rm  \
		--name aml-sdk_azcli_container \
		-p 9999:9999 \
		-v $(PWD):/workspace:rw \
		-v /usr/bin/docker:/usr/bin/docker \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-e DOCKER_REPOSITORY=$(dockerhub)
		-it $(image_name) \
		/bin/bash -c "jupyter notebook --notebook-dir=/workspace --ip=0.0.0.0 --port=9999 --no-browser --allow-root"

remove-notebook: 
	rm -f test.ipynb

clean: remove-notebook
	conda remove --name $(aml_sdk_conda_env_name) -y --all
	rm -rf .ipynb_checkpoints

.PHONY: help test setup clean remove-notebook test-notebook000 test-notebook010 test-notebook020
