# Democratizing LLM inference

## Ideas

system orchestration (figure out how and when to migrate workloads)  
workload management (e.g. how to serve 10k inference requests as best as possible; connecting/disconnecting clients; ...)  
device heterogeneity  
runtime adaptation based on connected device (e.g. different quantization if more clients)  
network topology (where are models stored, how to serve large chunks efficiently, what about bad network connections)

## Research questions

How to distribute load based given heterogeneous devices?  
How can you collect information from web-based clients to make workloadmanagement decisions?
