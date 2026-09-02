#pragma once

#include <typeinfo>
#include <pluginlib/class_loader.h>
#include <memory>
#include <sstream>
#include <vector>
#include "ros/assert.h"
#include "ros/console.h"
#include "ros/ros.h"

namespace terrain_analyzer
{
    typedef std::map<std::string, XmlRpc::XmlRpcValue> string_map_t;

    template<typename T>
    class PostProcessor
    {
    public:

        PostProcessor():configured_(false){};

        virtual ~PostProcessor(){};

        bool configure(const std::string& param_name, ros::NodeHandle node_handle = ros::NodeHandle())
        {
            XmlRpc::XmlRpcValue config;
            if (!node_handle.getParam(param_name, config))
            {
                ROS_ERROR("Could not find parameter %s on the server, are you sure that it was pushed up correctly?", param_name.c_str());
                return false;
            }
            return configure(config);
            
        }

        bool configure(XmlRpc::XmlRpcValue& config)
        {
            if (configured_)
            {
                ROS_WARN("PostProcessor %s of type %s already being reconfigured", post_processor_name_.c_str(), post_processor_type_.c_str());
            };
            configured_ = false;
            bool retval = true;

            retval = retval && loadConfiguration(config);
            retval = retval && configure();
            configured_ = retval;
            return retval;
        }

        virtual bool process(const T& data_in, T& data_out)=0;

        std::string getType() {return post_processor_type_;};

        const std::string& getName() const {return post_processor_name_;};

    protected:

        virtual bool configure()=0;

        bool getParam(const std::string& name, std::string& value) const
        {
            string_map_t::const_iterator it = params_.find(name);
            if (it == params_.end())
            {
                return false;
            }

            if(it->second.getType() != XmlRpc::XmlRpcValue::TypeString)
            {
                return false;
            }

            auto tmp = it->second;
            value = std::string(tmp);
            return true;
        }

        bool getParam(const std::string& name, bool& value) const
        {
            string_map_t::const_iterator it = params_.find(name);
            if (it == params_.end())
            {
                return false;
            }

            if(it->second.getType() != XmlRpc::XmlRpcValue::TypeBoolean)
            {
                return false;
            }

            auto tmp = it->second;
            value = (bool)(tmp);
            return true;
        }

        bool getParam(const std::string&name, double& value) const
        {
            string_map_t::const_iterator it = params_.find(name);
            if (it == params_.end())
            {
                return false;
            }

            if(it->second.getType() != XmlRpc::XmlRpcValue::TypeDouble && it->second.getType() != XmlRpc::XmlRpcValue::TypeInt)
            {
                return false;
            }

            auto tmp = it->second;
            value = it->second.getType() == XmlRpc::XmlRpcValue::TypeInt ? (int)(tmp) : (double)(tmp);
            return true;
        }

        bool getParam(const std::string&name, int& value) const
        {
            string_map_t::const_iterator it = params_.find(name);
            if (it == params_.end())
            {
                return false;
            }

            if(it->second.getType() != XmlRpc::XmlRpcValue::TypeInt)
            {
                return false;
            }

            auto tmp = it->second;
            value = tmp;
            return true;
        }

        bool getParam(const std::string&name, unsigned  int& value) const
        {
            int signed_value;
            if (!getParam(name, signed_value))
                return false;
            if (signed_value < 0)
                return false;
            value = signed_value;
            return true;
        };

        bool getParam(const std::string&name, std::vector<double>& value) const
        {
            string_map_t::const_iterator it = params_.find(name);
            if (it == params_.end())
            {
                return false;
            }

            value.clear();

            if(it->second.getType() != XmlRpc::XmlRpcValue::TypeArray)
            {
                return false;
            }

            XmlRpc::XmlRpcValue double_array = it->second;

            for (int i = 0; i < double_array.size(); ++i)
            {
                if(double_array[i].getType() != XmlRpc::XmlRpcValue::TypeDouble && double_array[i].getType() != XmlRpc::XmlRpcValue::TypeInt)
                {
                    return false;
                }

                double double_value = double_array[i].getType() == XmlRpc::XmlRpcValue::TypeInt ? (int)(double_array[i]) : (double)(double_array[i]);
                value.push_back(double_value);
            }
            
            return true;
        }

        bool getParam(const std::string&name, std::vector<std::string>& value) const
        {
            string_map_t::const_iterator it = params_.find(name);
            if (it == params_.end())
            {
                return false;
            }

            value.clear();

            if(it->second.getType() != XmlRpc::XmlRpcValue::TypeArray)
            {
                return false;
            }

            XmlRpc::XmlRpcValue string_array = it->second;
            
            for (unsigned int i = 0; i < string_array.size(); ++i)
            {
                if(string_array[i].getType() != XmlRpc::XmlRpcValue::TypeString)
                {
                    return false;
                }

                value.push_back(string_array[i]);
            }

            return true;
        }

        bool getParam(const std::string& name, XmlRpc::XmlRpcValue& value) const
        {
            string_map_t::const_iterator it = params_.find(name);
            if (it == params_.end())
            {
                return false;
            }

            auto tmp = it->second;
            value = tmp;
            return true;
        }
        
        std::string post_processor_name_;

        std::string post_processor_type_;
        
        bool configured_;

        string_map_t params_;

    private:

        bool setNameAndType(XmlRpc::XmlRpcValue& config)
        {
            if(!config.hasMember("name"))
            {
            ROS_ERROR("PostProcessor didn't have name defined, other strings are not allowed");
            return false;
            }

            std::string name = config["name"];

            if(!config.hasMember("type"))
            {
            ROS_ERROR("PostProcessor %s didn't have type defined, other strings are not allowed", name.c_str());
            return false;
            }

            std::string type = config["type"];

            post_processor_name_ = name;
            post_processor_type_ = type;
            ROS_DEBUG("Configuring PostProcessor of Type: %s with name %s", type.c_str(), name.c_str());
            return true;
        }

    protected:
        bool loadConfiguration(XmlRpc::XmlRpcValue& config)
        {
            if(config.getType() != XmlRpc::XmlRpcValue::TypeStruct)
            {
                ROS_ERROR("A PostProcessor configuration must be a map with fields name, type, and params");
                return false;
            } 

            if (!setNameAndType(config))
            {
                return false;
            }

            if(config.hasMember("params"))
            {
                XmlRpc::XmlRpcValue params = config["params"];

                if(params.getType() != XmlRpc::XmlRpcValue::TypeStruct)
                {
                    ROS_ERROR("params must be a map");
                    return false;
                }
                else
                {
                    for(XmlRpc::XmlRpcValue::iterator it = params.begin(); it != params.end(); ++it)
                    {
                        ROS_DEBUG("Loading param %s\n", it->first.c_str());
                        params_[it->first] = it->second;
                    } 
                }
            }

            return true;    
        }
    };

    template <typename T>
    class PostProcessorChain
    {
    private:
        pluginlib::ClassLoader<terrain_analyzer::PostProcessor<T>> loader_;
    public:
        
        PostProcessorChain(std::string data_type): loader_("terrain_analyzer", std::string("terrain_analyzer::PostProcessor<") + data_type + std::string(">")), configured_(false)
        {
            std::string lib_string = "";
            std::vector<std::string> libs = loader_.getDeclaredClasses();
            for (unsigned int i = 0 ; i < libs.size(); i ++)
            {
                lib_string = lib_string + std::string(", ") + libs[i];
            }    
            ROS_DEBUG("In PostProcessorChain ClassLoader found the following libs: %s", lib_string.c_str());
        }

        ~PostProcessorChain()
        {
            clear();
        }

        bool configure(std::string param_name, ros::NodeHandle node = ros::NodeHandle())
        {
            XmlRpc::XmlRpcValue config;
            if(node.getParam(param_name + "/filter_chain", config))
            {
                std::string resolved_name = node.resolveName(param_name).c_str();
                ROS_WARN("Filter chains no longer check implicit nested 'filter_chain' parameter.  This node is configured to look directly at '%s'.  Please move your chain description from '%s/filter_chain' to '%s'", resolved_name.c_str(), resolved_name.c_str(), resolved_name.c_str());
            }
            else if(!node.getParam(param_name, config))
            {
                ROS_DEBUG("Could not load the chain configuration from parameter %s, are you sure it was pushed to the parameter server? Assuming that you meant to leave it empty.", param_name.c_str());
                configured_ = true;
                return true;
            }
            return this->configure(config, node.getNamespace());
        }

        bool process(const T& data_in, T& data_out)
        {
            unsigned int list_size = reference_pointers_.size();
            bool result;
            if (list_size == 0)
            {
                data_out = data_in;
                result = true;
            }
            else if (list_size == 1)
                result = reference_pointers_[0]->process(data_in, data_out);
            else if (list_size == 2)
            {
                result = reference_pointers_[0]->process(data_in, buffer0_);
                if (result == false) {return false; }; //don't keep processing on failure
                    result = result && reference_pointers_[1]->process(buffer0_, data_out);
            }
            else
            {
                result = reference_pointers_[0]->process(data_in, buffer0_);  //first copy in
                for (unsigned int i = 1; i <  reference_pointers_.size() - 1; i++) // all but first and last (never called if size=2)
                {
                    if (i %2 == 1)
                        result = result && reference_pointers_[i]->process(buffer0_, buffer1_);
                    else
                        result = result && reference_pointers_[i]->process(buffer1_, buffer0_);
                    
                    if (result == false) {return false; }; //don't keep processing on failure
                }
                if (list_size % 2 == 1) // odd number last deposit was in buffer1
                    result = result && reference_pointers_.back()->process(buffer1_, data_out);
                else
                    result = result && reference_pointers_.back()->process(buffer0_, data_out);
            }
            return result;
            
        }

        bool clear() 
        {
            configured_ = false;
            reference_pointers_.clear();
            return true;
        }

        bool configure(XmlRpc::XmlRpcValue& config, const std::string& filter_ns)
        {
            if (config.getType() != XmlRpc::XmlRpcValue::TypeArray)
            {
                ROS_ERROR("%s: The PostProcessor chain specification must be a list. but is of of XmlRpcType %d", filter_ns.c_str(), config.getType());
                ROS_ERROR("The xml passed in is formatted as follows:\n %s", config.toXml().c_str());

                return false;
            }

            for (int i = 0; i < config.size(); ++i)
            {
                if(config[i].getType() != XmlRpc::XmlRpcValue::TypeStruct)
                {
                    ROS_ERROR("%s: PostProcessors must be specified as maps, but they are XmlRpcType:%d", filter_ns.c_str(), config[i].getType());
                    return false;
                }
                else if (!config[i].hasMember("type"))
                {
                    ROS_ERROR("%s: Could not add a PostProcessor because no type was given", filter_ns.c_str());
                    return false;
                }
                else if (!config[i].hasMember("name"))
                {
                    ROS_ERROR("%s: Could not add a PostProcessor because no name was given", filter_ns.c_str());
                    return false;
                }
                else
                {
                    for (int j = i + 1; j < config.size(); ++j)
                    {
                        if(config[j].getType() != XmlRpc::XmlRpcValue::TypeStruct)
                        {
                            ROS_ERROR("%s: PostProcessors must be specified as maps, but they are XmlRpcType:%d", filter_ns.c_str(), config[j].getType());
                            return false;
                        }

                        if(!config[j].hasMember("name")
                            ||config[i]["name"].getType() != XmlRpc::XmlRpcValue::TypeString
                            || config[j]["name"].getType() != XmlRpc::XmlRpcValue::TypeString)
                        {
                            ROS_ERROR("%s: PostProcessors names must be strings, but they are XmlRpcTypes:%d and %d", filter_ns.c_str(), config[i].getType(), config[j].getType());
                            return false;
                        }

                        std::string namei = config[i]["name"];
                        std::string namej = config[j]["name"];
                        if (namei == namej)
                        {
                            ROS_ERROR("%s: A self_filter with the name %s already exists", filter_ns.c_str(), namei.c_str());
                            return false;
                        }
                    }

                    if (std::string(config[i]["type"]).find("/") == std::string::npos)
                    {
                        ROS_ERROR("Bad PostProcessor type %s. Filter type must be of form <package_name>/<filter_name>", std::string(config[i]["type"]).c_str());
                        return false;
                    }
                    //Make sure the PostProcessor chain has a valid type
                    std::vector<std::string> libs = loader_.getDeclaredClasses();
                    bool found = false;
                    for (std::vector<std::string>::iterator it = libs.begin(); it != libs.end(); ++it)
                    {
                        if (*it == std::string(config[i]["type"]))
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        ROS_ERROR("Couldn't find PostProcessor of type %s", std::string(config[i]["type"]).c_str());
                        return false;
                    }
                }
            }
            
            bool result = true;    
            for (int i = 0; i < config.size(); ++i)
            {
                auto p(loader_.createUnmanagedInstance(config[i]["type"]));
                if (p == nullptr)
                    return false;
                std::shared_ptr<terrain_analyzer::PostProcessor<T>> ptr(p);
                result = result &&  ptr->configure(config[i]);    
                reference_pointers_.push_back(ptr);
                std::string type = config[i]["type"];
                std::string name = config[i]["name"];
                ROS_DEBUG("%s: Configured %s:%s PostProcessor at %p\n", filter_ns.c_str(), type.c_str(), name.c_str(),  p);
            }
            
            if (result == true)
            {
                configured_ = true;
            }
            return result;
        }

        std::vector<std::shared_ptr<terrain_analyzer::PostProcessor<T>>> getFilters() const
        {
            return reference_pointers_;
        }

    private:

        std::vector<std::shared_ptr<terrain_analyzer::PostProcessor<T>>> reference_pointers_;

        T buffer0_;
        T buffer1_;
        bool configured_;
    };
}
