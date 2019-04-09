# Operationalizing R models using Python via Azure ML SDK

## _Note: This work is still in the testing phase._

This is a tutorial on how to use [Azure Machine Learning SDK for Python](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py) (__AML SDK__) to operationalize (Figure 1) pre-trained R models at scale in the cloud via [Azure Kubernetes Service](https://docs.microsoft.com/en-us/azure/aks/). While R models' operationalization (__o16n__) is not the focus of the current AML SDK release, we show here how to use [conda](https://conda.io/en/latest/)  to [install](https://docs.anaconda.com/anaconda/user-guide/tasks/use-r-language/) [R](https://anaconda.org/r/r-base) and R packages, and how to leverage the [rpy2](https://rpy2.bitbucket.io/) Python package to start and interact with an R session embedded in a Python process. 

We focus here on standard (i.e. not deep learning) machine learning (__ML__) algorithms using a Kernel SVM based R model as an example. This solution can be generalized for any pre-trained R model with a corresponding R scoring script. 

![Experimentation and Orchestration:](https://user-images.githubusercontent.com/16708375/54651754-a377f200-4a8a-11e9-85fe-bb76b3b40e66.png "Experimentation and Orchestration")
Figure 1. Code structure for R models o16n via Python and AML SDK:
1. [AML SDK model management](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-model-management-and-deployment) allows transparent tracking and versioning of any model in any model serialization file format, including R `.rds` files (in addition to Python `.pkl`).
2. In AML SDK, a Python o16n script must supply two functions:
    * `init()` is run once when the deployment service is created. This initialization function will start the R session and load the R model.
    * `run()` is called every time the service is used to score new data. It is used here to pass data to the R session created in the `init()` function, call the R scoring function, and return the results to Python which will return the service results back to calling app.
3. The conda env file defines the  o16n Python environment and it provides a transparent encapsulation of all o16n R and Python packages that are added on top of the o16n base docker image. For R models o16n, we install the required Python packages (`rpy2`) and R ([r-base](https://anaconda.org/r/r-base)), as well as any required R libraries, like [kernlab](https://anaconda.org/conda-forge/r-kernlab) for example. 
4. AML SDK o16n base docker image is predefined at present, but "bring your own base" (__BYOB__) docker image will be available soon. For R models o16n the base docker image requirements are not very complex since we just need `rpy2` and the ability to load and run our R model.  
    
#### AI solutions development pipeline in AML SDK
The Azure ML application development process (Figure 2) connects the data scientist's Windows or Linux orchestration machine (a local computer or an Azure Virtual Machine (VM)) to an Azure __Ubuntu__ VM used as a [compute target](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets) to run distinct docker containers for each of the following four fundamental [stages](https://user-images.githubusercontent.com/16708375/48814903-90d2f380-ed0a-11e8-8f4e-928171dea7dc.png):
  1. Orchestration: This is covered by AML SDK, which can be  [installed](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/install?view=azure-ml-py)  by the user, or used directly inside a docker container that runs a [simple prebuilt AML SDK docker image](https://github.com/georgeAccnt-GH/PowerAIWithDocker/blob/master/amlsdk/docker_history/dockerfile_aml-sdk_docker_image_sdk.v.1.0.17). For simplicity the latter option is used here. The orchestration docker [container](https://hub.docker.com/r/georgedockeraccount/aml-sdk_docker_image/tags) is started manually on either a local (Windows or Linux) machine, or on an Azure VM, or directly on an Ubuntu Azure VM compute target (in which case the compute target machine is the same as the one used for orchestration). 

 2. ML model training i.e. experimentation: This part is assumed to have been done outside Azure, for example using a tool like R Studio to train a ML model and develop the associated R scoring script.
 
 3. Development of ML scoring script:  AML SDK experimentation framework usage is extended here beyond ML model training for developing the docker image running the o16n python script that invokes the user provided R scoring script to load a serialized pretrained R model and apply to score new data. This step is optional in this tutorial but is useful for generic dockerized scripts development and is further described below.

 4. o16n: AML SDK is used to deploy the R model on an Azure Container Instance [ACI](https://docs.microsoft.com/en-us/azure/container-instances/) for testing the Azure o16n deployment stack (flask app) and finally to an Azure Kubernetes Service [AKS] cluster.  



![Architecture for building an AI pipeline:](https://user-images.githubusercontent.com/16708375/54639528-0013e680-4a64-11e9-8b9b-13ddc79c20b6.png "Architecture for building an AI pipeline:")
Figure 2. Architecture for building an AI pipeline:
 1. Data is ingested for training.
 2. An __ML__ model is trained through an experimentation process.
 3. A Model and the associated scoring script are packaged in a [Docker](https://www.docker.com/) scoring image that can be stored in a [Azure Container Registry (ACR)](https://docs.microsoft.com/en-us/azure/container-registry/).
 4. The scoring image is deployed (operationalized) on an [Azure Container Instance (ACI)](https://docs.microsoft.com/en-us/azure/container-instances/) for small deployments or on an [Azure Kubernetes Service (AKS)](https://docs.microsoft.com/en-us/azure/aks/) cluster for inference at scale.
 5. New observation data can be scored in both real time or batch modes.
 6. Model scores can be used or stored for AI consumption.

#### Development of containerized scripts using AML SDK 

AML SDK has a powerful experimentation infrastructure (Figure 3), which is tuned for ML model training and o16n in python, but can also be leveraged for R models o16n script development.

![Experimentation and Orchestration:](https://user-images.githubusercontent.com/16708375/54651601-fe5d1980-4a89-11e9-8617-be4ac90c3c02.png "Experimentation and Orchestration")
Figure 2. Experimentation framework in AML SDK:  
1. Orchestration machine: The data scientist's working location, shown on upper right can be a local Windows or Linux computer or a cloud based VM. This machine will run the AML SDK either locally or in a docker container.
2. The  [compute target](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets) is an Azure __Ubuntu__ VM used to develop and run docker containers, either generic or specific for ML model training. The AML SDK will be used to create the training and scoring scripts, package dependencies (in a conda environment .yml file) and a docker base image for scoring script development.
3. The same code structure (scoring script, dependencies, base docker image) is used to create the o16n docker image, which can be deployed to either an [ACI](https://docs.microsoft.com/en-us/azure/container-instances/) or an [AKS](https://docs.microsoft.com/en-us/azure/aks/) cluster for model inference.  


The o16n script is written in python, but creates an internal R session via rpy2. The session runs the user provided R scoring script. R models are accessibile using simple interactions with the R session (Figure 1): load the serialized model (once when scoring service is created), pass scoring data, and return model results.


## Prerequisites
Both the Orchestration and the AML SDK Compute Target machines can be deployed and configured either via [Azure portal](https://portal.azure.com) or via [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest).  
> Full [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) instructions are provided below for deploying and configuring (i.e. opening up ports for ssh and Jupyter server connections) of an Azure VM.

 __NOTE__: Securing access to VMs and to the notebook server is paramount, but outside the scope of this tutorial. We use here simple security measures like ssh via login and password using non-standard ports. More secure ways use [ssh keys](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/ssh-from-windows) and access control list [(ACL) endpoints](https://docs.microsoft.com/en-us/previous-versions/azure/virtual-machines/linux/classic/setup-endpoints). It is highly recommended to address the access security issue before starting an AI development project. 

* Orchestration machine: 
   - Linux or Windows local computer or [Azure virtual machine](https://docs.microsoft.com/en-us/azure/virtual-machines/) for running the "control panel" notebooks. Linux (e.g. Ubuntu Azure [DSVM](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro)) is easier to use if you wish to test the python scoring script outside AML SDK because it allows creating dockerized scripts that have linux base docker images like [continuumio/miniconda3:4.5.12](https://github.com/ContinuumIO/docker-images/tree/master/miniconda3) used in this project.
   - [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) installed
   - [Dockerhub](https://hub.docker.com/) account (but see [ACR](https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-azure-cli) for an Azure secure alternative)
   - If the orchestration machine is an Azure VM:
     - [open](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/nsg-quickstart-portal) up one port for Jupyter Notebook server (that will run inside the orchestration docker container). 
     - opening an ssh port may also be useful for connecting to an Azure __Linux__ VM  orchestration machine. Remote desktop protocol [RDP](https://docs.microsoft.com/en-us/azure/marketplace/cloud-partner-portal/virtual-machine/cpp-connect-vm) can be used for Azure __Windows__ VMs. 
   - If the orchestration machine is a local computer, port 8888 can be used for Jupyter Notebook, and ssh may not be needed. 
   
   See Azure CLI commands below for deploying an Azure VM and for opening up ports for ssh and Jupyter Server connections. 
    
*  AML SDK Compute Target machine:  

     * Deploy an Azure [ Ubuntu Data Science Virtual Machine (DSVM)](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro) to be used as a [compute target](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets) for testing the scoring scripts (see below instructions on how to do the DSVM deployment and configuration using CLI). Azure VMs are a subset of the [compute targets](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets#compute-targets-for-training) available in AML SDK, and they can be either managed (provisioned, scaled, used and re-used, and destroyed as needed by SDK) or un-managed, i.e. by allowing the users to bring their own compute resource provisioned outside SDK. For simplicity, we will use a not-managed VM here, although the demo can be changed easily to leverage SDK managed compute resources.  
    * Beside DSVM, other Linux VM [sizes](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/sizes) will also work provided they have docker and Jupyter notebook server installed. Although not compulsory for the simple example shown here, we are using a [GPU](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu) enabled VM (Standard_NC6), which is usually a must for compute intensive tasks.   
    * On the deployed AML SDK compute target machine, [open](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/nsg-quickstart-portal) up one port for ssh connections to be used by SDK to submit computing jobs (this port can be shared with user created ssh sessions). 
    
   See Azure CLI commands below for deploying an Azure VM and for opening up ports for ssh and Jupyter Server connections.  

 * #### VM provisioning and configuration via [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest):
  
    As an alternative to [provisioning](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro) and [configuration](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/nsg-quickstart-portal) instructions using Azure portal, here are the  [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) commands to deploy a large sized disk Ubuntu DSVM, and open two ports for ssh and two ports for Jupyter notebook using an Azure [Network Security Group (nsg)](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/nsg-quickstart). 

    Instructions for running the VM provisioning and configuration Azure CLI script:
    
    *  run subscription setting commands (__az login__, __az account list ...__, and  __az account set ...__ ) separately, and make sure the right Azure subscription is [set](https://docs.microsoft.com/en-us/cli/azure/manage-azure-subscriptions-azure-cli?view=azure-cli-latest) as active.
    
    *  set __prefix__ variable to a small caps string that will be the base name driving all Azure resources names provisioned here (e.g. test01). It must conform to the following regular expression: \^[a-z][a-z0-9-]{1,61}[a-z0-9]$.
    
    *  set the values for __VM_ADMIN_PWD__ and __VM_ADMIN_LOGIN__. 

    *  change __location__, __customsshPorts__ and __customJupyterPorts__ ports as needed (space separated values).  

    *  record values used for next variables (they will be used in the project notebooks): 
        *  __VM_ADMIN_PWD__, __VM_ADMIN_LOGIN__
        *  __customsshPorts__, and __customJupyterPorts__ 
        *  __location__ and __rsgname__
        *  __public-ip-address-dns-name__ parameter (should be __%prefix%testvm__)
        Last three sets of variables can also be found using the Azure portal.  

    *  the script will run on a machine with Azure CLI installed or inside the SDK linux docker image container used in this project:
        - Windows: in Windows Command Prompt or a similar application like Anaconda prompt. 
        - Linux: 'SET' key word should be removed (or replaced with 'ENV'), while variable calls should use "$variable" syntax rather than "%variable%" (see  `code/amlsdk_operationalization/AzureVMviaAzureCLI.sh`).

        - typical Anaconda prompt session on a Windows machine running the SDK linux docker image container:
            ```
            (base) C:\repos\repoDirOnWinMachine>docker run --rm --name testAZCLIInSDKDocker -it -v %cd%:/workspace georgedockeraccount/aml-sdk_docker_image:sdk.v.1.0.21 /bin/bash
            (base) root@31767d66325d:/# az --version
                azure-cli                         2.0.61
            (base) root@31767d66325d:/# az login
                To sign in, use a web browser to open the page https://microsoft.com/devicelogin and enter the code **** to authenticate.
            (base) root@31767d66325d:/# sh /workspace/code/amlsdk_operationalization/AzureVMviaAzureCLI.sh
            ```
    *  when the script is finished:
        *  use the Azure portal to navigate to newly created rsg (defined by variable __rsgname__), open the vm blade and check the __Networking__ settings to see that the  __customsshPorts__ and __customJupyterPorts__ ports are now open.
       *  see the next set of instructions below the script on how to update the ssh service on the Linux VM to use the newly opened __customsshPorts__. 

```
ECHO:"VM provisioning and configuration via Azure CLI!"
ECHO:"Run 'az login' before running this script, and set the right subscription."
: # az login
: # az account list --all --refresh -o table 
: # az account set --subscription "subscr id or name"

ECHO:"Make sure the variables in block below are properly defined."   
: # user defined variables
: # replace everything inside block below, including the brackets []
: #---- BEGIN user defined variables
: # Prefix must be all lowercase, no caps.
SET prefix=[change this->]prefix01
SET location="eastus"
SET customsshPorts=[change this->]9555 9556
SET customJupyterPorts=[change this->]9666 9667
SET VM_ADMIN_PWD=
SET VM_ADMIN_LOGIN=
: #---- END user defined variables

echo %customJupyterPorts%
echo %customsshPorts%

: # set resource group name (rsg)
SET basename=%prefix%_tst_01_
SET rsgname=%basename%rsg
echo %rsgname%

: # create rsg and provision a GPU enabled (Standard_NC6) Ubuntu DSVM with a 300GB disk, and a reserved public-ip-address-dns-name (FQDN)
az group create --name %rsgname% --location %location%
az vm create -n %prefix%testvm -g %rsgname% --image microsoft-dsvm:linux-data-science-vm-ubuntu:linuxdsvmubuntu:latest  --admin-password %VM_ADMIN_PWD% --admin-username %VM_ADMIN_LOGIN% --size Standard_NC6 --public-ip-address-dns-name %prefix%testvm --os-disk-size-gb 300

: # create a network security group (NSG), to which we later add two network inbound rules that open customsshPorts and customJupyterPorts
az network nsg create --resource-group %rsgname% --location %location% --name %basename%nsg

: # create rule that opens up customsshPorts
az network nsg rule create --resource-group %rsgname% --nsg-name %basename%nsg --name %basename%nsg_rule01 --protocol tcp --priority 1010 --destination-port-ranges %customsshPorts% --direction Inbound  --access Allow --description "custom ssh"

: # create rule that opens up customJupyterPorts
az network nsg rule create --resource-group %rsgname% --nsg-name %basename%nsg --name %basename%nsg_rule02 --protocol * --priority 1020 --destination-port-ranges %customJupyterPorts% --direction Inbound  --access Allow --description "custom jupyter"

: # typically, the deployed DSVM network interface name (NIC) will be %prefix%testvmVMNIC, but you can check this before you add the nsg to Vm's NIC
az vm nic list --resource-group %rsgname% --vm-name %prefix%testvm

: # Apply the newly created nsg to DSVM
az network nic update --resource-group %rsgname% --name %prefix%testvmVMNIC --network-security-group %basename%nsg

: # Create a storage account, that could be used if 'Boot Diagnostics' are enabled on Portal, and thus serical console is available.
az storage account create -n %prefix%sa -g %rsgname% --location %location% --sku Standard_LRS

: # Enable boot diagnostic so that serical console access to VM is enabled
az vm boot-diagnostics enable --name %prefix%testvm --resource-group %rsgname% --storage "https://%prefix%sa.blob.core.windows.net/"

: #  clean up
: #  az group delete -n %rsgname% --no-wait --yes
```
  
 * #### SSH service update on an Azure Linux VM:

    The last configuration step is to [change the default ssh ports](https://www.godaddy.com/help/changing-the-ssh-port-for-your-linux-server-7306) on the linux VM, by editing file __etc/ssh/sshd_config__ to add __customsshPorts__ used in the script above and remove port 22, and then restarting the sshd service via __service sshd restart__ command. 
    
    Typically, default ssh port 22 is closed soon after the vm is provisioned, and the newly opened __customsshPorts__ are not yet useful, because the ssh server on the VM still listens to default port. If you can not ssh into the vm using port 22, you can still access the vm using this approach:  
            *  navigate to Azure portal  
            *  locate the VM rsg, and select the VM  
            *  enable __Boot diagnostics__ in the __Support + troubleshooting__ settings of the VM (you can create a new Storage account in the same rsg as the VM).  
            *  go to __Serial console__ in the __Support + troubleshooting__ settings of the VM.   

    ##### To enable SSH access into Ubuntu VMs  
     * ssh into the vm using either port 22 or the serial console in the portal.
     * use the credentials (__VM_ADMIN_PWD__, __VM_ADMIN_LOGIN__ variables) used in the provisioning AZ CLI script, and you should have command line access to the VM
     * change [ssh ports](https://www.godaddy.com/help/changing-the-ssh-port-for-your-linux-server-7306) in __/etc/ssh/sshd_config__ file to match the values used for __customsshPorts__ script variable only (i.e. not the __customJupyterPorts__). Use __sudo__ if needed. 
     * restart sshd service (by running __service sshd restart__). 
     * restart the VM (using 'Restart' button on VM portal main blade)

  
As described above, AML SDK [installation](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/install?view=azure-ml-py) on the Windows or Linux orchestration machine is not used here. Instead the AML SDK is used directly in a docker container running a [transparently built](https://github.com/georgeAccnt-GH/PowerAIWithDocker/blob/master/amlsdk/docker_history/dockerfile_aml-sdk_docker_image_sdk.v.1.0.17) docker [image](https://hub.docker.com/r/georgedockeraccount/aml-sdk_docker_image/tags).

   
## Setup

Use orchestration machine:
1. Clone this repo
2. `cd` into the repo directory.  
3. Make sure you are logged into your dockerhub account by running:

```
docker login 
```
4. This project notebooks can be run in an AML SDK container on windows or linux orchestration machine:  
> __on a windows local machine (for example using an Anaconda prompt)__:
    Start the AML SDK  __Linux__ docker container using command below (with [VISIBLE_PORT] being a port of your choice, for example 8889, or even 8888, while INSIDE_PORT is usually 8888)  :
```
docker run -it -p [VISIBLE_PORT]:[INSIDE_PORT] -v %cd%:/workspace georgedockeraccount/aml-sdk_docker_image:sdk.v.1.0.21 /bin/bash -c "jupyter notebook --notebook-dir=/workspace --ip=0.0.0.0 --port=[INSIDE_PORT] --no-browser --allow-root"  

```   
After Jupiter server starts, it will display a security token that should be used as shown below, and port [INSIDE_PORT] opened inside the container which should __not__ be used. Use [__VISIBLE_PORT__] value instead of [INSIDE_PORT] displayed by Jupiter server running in SDK Docker container. Point a local browser on the orchestration machine to:  
```
http://localhost:[VISIBLE_PORT]/?token=securitytoken_printed_by_jupyter_session_started_above

```
> __on an Azure Linux VM named [yourVM]__:
    ssh into [yourVM], do the three setup steps described above (repo cloning, cd into repo directory and check dockerhub connection) and then start the AML SDK __Linux__ docker container using command below (with [VISIBLE_PORT] being any of the ports opened for Jupyter server when the orchestration VM was provisioned, for example by opening the __customJupyterPorts__ in the AZ CLI script above), 
```
[your_login_info]@[yourVM]:/repos/AMLSDKOperationalizationRModels$ docker run -it -p [VISIBLE_PORT]:[INSIDE_PORT] -v ${PWD}:/workspace:rw georgedockeraccount/aml-sdk_docker_image:sdk.v.1.0.21 /bin/bash -c "jupyter notebook --notebook-dir=/workspace --ip=0.0.0.0 --port=[INSIDE_PORT] --no-browser --allow-root"  

```   
After Jupiter server starts, it will display a security token that should be used below, and port [INSIDE_PORT] opened inside the container which should __not__ be used. Use [__VISIBLE_PORT__] value instead of [INSIDE_PORT] displayed by Jupiter server running in SDK Docker container. Point local (on any machine connected to the internet, including the orchestration machine) browser to:  
```
[yourVM].[yourRegion].cloudapp.azure.com:[VISIBLE_PORT]/?token=securitytoken_printed_by_jupyter_session_started_above  

```

5. Once the Jupyter notebook is started, open and run the notebooks in __/code/amlsdk_operationalization/__ in this order:
 
  * `000_RegularR_RealTime_Scripts_and_SDK_setup.ipynb`: the notebook is used as an IDE to write utility scripts to disk. It also sets up AML SDK infra-structure: authorization, Azure resources (resource groups, AML workspaces) provisioning. Significant steps:
    * Locate notebook cell that contains sensitive information, and replace the existing empty python dictionary variable __sensitive_info__ with the required information: Azure subscription ID, AML SDK Compute Target machine login and configuration info. 
    * You should do the step described above only once, and thus save the info in the untracked .env file. 
    * Although setting up Service principal is optional, using the default values for Service Principal information in the sensitive info dictionary is __NOT__ optional. The get_auth() utility function defined in the notebook and saved in o16n_regular_ML_R_models_utils.py checks if Service Principal password has to default value ("YOUR_SERVICE_PRINCIPAL_PASSWORD") to switch to interactive Azure login.

  * `010_RegularR_RealTime_test_score_in_docker.ipynb`: __optional__ notebook that shows how the AML SDK experimentation framework can be used to develop generic (i.e. not necesarily for ML training) containerized scripts. Significant steps:
    * R model registration. Even if AML SDK is focused on python, any file can be registered and versioned. We __can__ access registered models even when using the AML SDK experimentation framework.
    * We use an already provisioned VM as an AML SDK un-managed [compute target](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets). As mentioned above, it is possible to use the same Linux VM as both a compute target machine and orchestration machine.
    * the script we develop is written an python. It creates an R session in function init() (which will be invoked once at o16n time in next notebook, when service is deployed) and loads the R model in R session memory. The R session is used for scoring in run() function. Besides R model loading at service creation time and regular scoring of new data, the scripts also report processing and data passing (python to R adn back) times. Results show the data passing times are relatively constant as a function of data size (10 ms) so the data transfer overhead is minimal especially for large data sets. 

 
  * `020_RegularR_RealTime_deploi_ACI_AKS.ipynb`: deploys the above o16n script and R model on an ACI and AKS. It creates an o16n image which can be pulled from its ACR and tested locally if needed using the SDK or the portal. The notebook also provisions and ACI for remote testing and finally and AKS cluster where the o16n image can be used to score data in real time (i.e. not batch processing) at scale.

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
