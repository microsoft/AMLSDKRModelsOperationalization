echo "VM provisioning and configuration via Azure CLI!"
echo "Run 'az login' before running this script, and set the right subscription."
: # az login
: # az account list --all --refresh -o table 
: # az account set --subscription "subscr id or name"
: # 
echo "Make sure the variables in block below are properly defined."
: #    
: # user defined variables
: # replace everything inside block below, including the brackets []
: #---- BEGIN user defined variables
prefix='ghiordanro16n02' # Prefix must be all lowercase, no caps.
location='eastus'
customsshPorts='9555-9556'
customJupyterPorts='9666-9667'
VM_ADMIN_PWD='pwd_VM__001-5*'
VM_ADMIN_LOGIN='loginvm001'
: #---- END user defined variables
: # 
echo "$customJupyterPorts"
echo "$customsshPorts"
: # 
: # set resource group name (rsg)
basename="${prefix}_tst_01_"
rsgname="${basename}rsg"
echo "$basename"
echo "$rsgname"
echo "$location"
: # 
: # create rsg and provision a GPU enabled (Standard_NC6) Ubuntu DSVM with a 300GB disk, and a reserved public-ip-address-dns-name (FQDN)
az group create --name "$rsgname" --location "$location"
az vm create -n "${prefix}testvm" -g "$rsgname" --image microsoft-dsvm:linux-data-science-vm-ubuntu:linuxdsvmubuntu:latest  --admin-password "$VM_ADMIN_PWD" --admin-username "$VM_ADMIN_LOGIN" --size Standard_NC6 --public-ip-address-dns-name "${prefix}testvm" --os-disk-size-gb 300
: # 
: # create a network security group (NSG), to which we later add two network inbound rules that open customsshPorts and customJupyterPorts
az network nsg create --resource-group "$rsgname" --location "$location" --name "${basename}nsg"
: # 
: # create rule that opens up customsshPorts
az network nsg rule create --resource-group "$rsgname" --nsg-name "${basename}nsg" --name "${basename}nsg_rule01" --protocol tcp --priority 1010 --destination-port-ranges "$customsshPorts" --direction Inbound  --access Allow --description "custom ssh"
: # 
: # create rule that opens up customJupyterPorts
az network nsg rule create --resource-group "$rsgname" --nsg-name "${basename}nsg" --name "${basename}nsg_rule02" --protocol '*' --priority 1020 --destination-port-ranges "$customJupyterPorts" --direction Inbound  --access Allow --description "custom jupyter"
: # 
: # typically, the deployed DSVM network interface name (NIC) will be "${prefix}testvmVMNIC", but you can check this before you add the nsg to Vm's NIC
az vm nic list --resource-group "$rsgname" --vm-name "${prefix}testvm"
: # 
: # Apply the newly created nsg to DSVM
az network nic update --resource-group "$rsgname" --name "${prefix}testvmVMNIC" --network-security-group "${basename}nsg"
: # 
: # Create a storage account, that could be used if 'Boot Diagnostics' are enabled on Portal, and thus serical console is available.
az storage account create -n "${prefix}sa" -g "$rsgname" --location "$location" --sku Standard_LRS
: #
: # Enable boot diagnostic so that serical console access to VM is enabled
az vm boot-diagnostics enable --name "${prefix}testvm" --resource-group "$rsgname" --storage "https://${prefix}sa.blob.core.windows.net/"
: # 
: #  clean up
: #  az group delete -n "$rsgname" --no-wait --yes
