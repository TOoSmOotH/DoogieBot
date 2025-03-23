import { useEffect, useState, useCallback } from 'react';
import AdminLayout from '../../components/layout/AdminLayout';
import { Button } from '../../components/ui/Button';
import { Card } from '../../components/ui/Card';
import withAdmin from '@/utils/withAdmin';
import {
  getLLMProviders,
  getAllLLMConfigs,
  createLLMConfig,
  updateLLMConfig,
  activateLLMConfig,
  deleteLLMConfig,
  getProviderModels,
} from '@/services/llm';
import { LLMConfig } from '@/types';

// Define the provider interface
interface Provider {
  id: string;
  name: string;
  available: boolean;
  default_model: string;
  requires_api_key: boolean;
  requires_base_url: boolean;
  available_models?: string[];
  embedding_models?: string[];
}

// Define the provider configuration interface
interface ProviderConfig {
  id: string;
  config: LLMConfig | null;
  enabled: boolean;
  api_key: string;
  base_url: string;
  model: string;
  embedding_model: string;
  system_prompt: string;
  available_models: string[];
  embedding_models: string[];
  isPolling: boolean;
}

const LLMConfiguration = () => {
  const [providers, setProviders] = useState<Record<string, Provider>>({});
  const [providerConfigs, setProviderConfigs] = useState<ProviderConfig[]>([]);
  const [configs, setConfigs] = useState<LLMConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [systemPrompt, setSystemPrompt] = useState('You are Doogie, a helpful AI assistant.');

  // Function to load data from the backend
  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      // Load providers
      const providersResponse = await getLLMProviders();
      if (providersResponse.error) {
        throw new Error(providersResponse.error);
      }

      const providersData = providersResponse.providers || {};
      setProviders(providersData);

      // Load configurations
      const configsResponse = await getAllLLMConfigs();
      if (configsResponse.error) {
        throw new Error(configsResponse.error);
      }

      const configsData = configsResponse.configs || [];
      setConfigs(configsData);

      // Initialize provider configs
      const initialProviderConfigs = Object.entries(providersData).map(
        ([id, info]: [string, any]) => {
          const existingConfig = configsData.find((c) => c.provider === id);
          return {
            id,
            config: existingConfig || null,
            enabled: existingConfig?.is_active || false,
            api_key: existingConfig?.api_key || '',
            base_url: existingConfig?.base_url || '',
            model: existingConfig?.model || info.default_model || '',
            embedding_model: existingConfig?.embedding_model || '',
            system_prompt:
              existingConfig?.system_prompt || 'You are Doogie, a helpful AI assistant.',
            available_models: [],
            embedding_models: [],
            isPolling: false,
          };
        }
      );

      setProviderConfigs(initialProviderConfigs);

      // Update systemPrompt state with the active configuration's system prompt
      const activeConfig = configsData.find((c) => c.is_active);
      if (activeConfig && activeConfig.system_prompt) {
        setSystemPrompt(activeConfig.system_prompt);
      }
    } catch (err) {
      console.error('Failed to load LLM data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load LLM data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  // Function to toggle a provider's enabled state
  const toggleProvider = async (providerId: string) => {
    // Find the provider config
    const providerConfig = providerConfigs.find((pc) => pc.id === providerId);
    if (!providerConfig) return;

    try {
      // If enabling, create or activate the config
      if (!providerConfig.enabled) {
        // If config exists, activate it
        if (providerConfig.config) {
          const response = await activateLLMConfig(providerConfig.config.id);
          if (response.error) {
            throw new Error(response.error);
          }
        } else {
          // Create new config
          const newConfig = {
            provider: providerId,
            model: providerConfig.model,
            embedding_model: providerConfig.embedding_model,
            system_prompt: systemPrompt, // Use the current system prompt value
            api_key: providerConfig.api_key,
            base_url: providerConfig.base_url,
          };

          const response = await createLLMConfig(newConfig);
          if (response.error) {
            throw new Error(response.error);
          }

          // Activate the new config
          if (response.config) {
            await activateLLMConfig(response.config.id);
          }
        }
      } else {
        // If disabling and another provider is available, activate that one
        const otherProvider = providerConfigs.find((pc) => pc.id !== providerId && pc.config);
        if (otherProvider && otherProvider.config) {
          await activateLLMConfig(otherProvider.config.id);
        }
      }

      // Reload data to reflect changes
      await loadData();
    } catch (err) {
      console.error('Failed to toggle provider:', err);
      setError(err instanceof Error ? err.message : 'Failed to toggle provider');
    }
  };

  // Function to update API key (local state only)
  const updateApiKey = (providerId: string, apiKey: string) => {
    // Update local state only
    setProviderConfigs((prevConfigs) =>
      prevConfigs.map((pc) => (pc.id === providerId ? { ...pc, api_key: apiKey } : pc))
    );
  };

  // Function to update base URL (local state only)
  const updateBaseUrl = (providerId: string, baseUrl: string) => {
    // Update local state only
    setProviderConfigs((prevConfigs) =>
      prevConfigs.map((pc) => (pc.id === providerId ? { ...pc, base_url: baseUrl } : pc))
    );
  };

  // Function to update model (local state only)
  const updateModel = (providerId: string, model: string) => {
    // Update local state only
    setProviderConfigs((prevConfigs) =>
      prevConfigs.map((pc) => (pc.id === providerId ? { ...pc, model } : pc))
    );
  };

  // Function to update embedding model (local state only)
  const updateEmbeddingModel = (providerId: string, embedding_model: string) => {
    // Update local state only
    setProviderConfigs((prevConfigs) =>
      prevConfigs.map((pc) => (pc.id === providerId ? { ...pc, embedding_model } : pc))
    );
  };

  // Function to save provider configuration
  const saveProviderConfig = async (providerId: string) => {
    // Find the provider config
    const providerConfig = providerConfigs.find((pc) => pc.id === providerId);
    if (!providerConfig) return;

    try {
      // If config exists, update it
      if (providerConfig.config) {
        // First update the config
        const response = await updateLLMConfig(providerConfig.config.id, {
          api_key: providerConfig.api_key,
          base_url: providerConfig.base_url,
          model: providerConfig.model,
          embedding_model: providerConfig.embedding_model,
        });

        if (response.error) {
          throw new Error(response.error);
        }

        // Then activate the config to make it take effect immediately
        await activateLLMConfig(providerConfig.config.id);

        setError(null);
        // Reload data to reflect changes
        await loadData();
        // Show success message after data is reloaded
        alert('Configuration saved successfully!');
      } else {
        // Create new config
        const newConfig = {
          provider: providerId,
          model: providerConfig.model,
          embedding_model: providerConfig.embedding_model,
          system_prompt: systemPrompt, // Use the current system prompt value
          api_key: providerConfig.api_key,
          base_url: providerConfig.base_url,
        };

        // First create the new config
        const response = await createLLMConfig(newConfig);
        if (response.error) {
          throw new Error(response.error);
        }

        if (response.config) {
          // Then activate the new config
          await activateLLMConfig(response.config.id);
        }

        setError(null);
        // Reload data to reflect changes
        await loadData();
        // Show success message after data is reloaded
        alert('Configuration created successfully!');
      }
    } catch (err) {
      console.error('Failed to save configuration:', err);
      setError(err instanceof Error ? err.message : 'Failed to save configuration');
    }
  };

  // Function to update system prompt
  const updateSystemPrompt = async () => {
    // Find the active provider config
    const activeProviderConfig = providerConfigs.find((pc) => pc.enabled);
    if (!activeProviderConfig || !activeProviderConfig.config) return;

    try {
      // Update the config
      const response = await updateLLMConfig(activeProviderConfig.config.id, {
        system_prompt: systemPrompt,
      });
      if (response.error) {
        throw new Error(response.error);
      }

      setError(null);
      // Reload data to reflect changes
      await loadData();
      // Show success message after data is reloaded
      alert('System prompt updated successfully!');
    } catch (err) {
      console.error('Failed to update system prompt:', err);
      setError(err instanceof Error ? err.message : 'Failed to update system prompt');
    }
  };

  // Function to poll for available models
  const pollAvailableModels = async (providerId: string) => {
    // Find the provider config
    const providerConfig = providerConfigs.find((pc) => pc.id === providerId);
    if (!providerConfig) return;

    // Set polling state
    setProviderConfigs((prevConfigs) =>
      prevConfigs.map((pc) => (pc.id === providerId ? { ...pc, isPolling: true } : pc))
    );

    try {
      // Call the API to get available models
      const response = await getProviderModels(
        providerId,
        providerConfig.api_key,
        providerConfig.base_url
      );

      if (response.error) {
        throw new Error(response.error);
      }

      // Update provider config with available models
      setProviderConfigs((prevConfigs) =>
        prevConfigs.map((pc) =>
          pc.id === providerId
            ? {
                ...pc,
                available_models: response.chatModels || [],
                embedding_models: response.embeddingModels || [],
                isPolling: false,
              }
            : pc
        )
      );
    } catch (err) {
      console.error('Failed to poll available models:', err);
      setError(err instanceof Error ? err.message : 'Failed to poll available models');

      // Reset polling state
      setProviderConfigs((prevConfigs) =>
        prevConfigs.map((pc) => (pc.id === providerId ? { ...pc, isPolling: false } : pc))
      );
    }
  };

  return (
    <AdminLayout title="LLM Configuration" description="Configure LLM providers and models">
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">LLM Configuration</h1>
        </div>

        {error && (
          <div className="p-4 bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-800 text-red-700 dark:text-red-400 rounded-md">
            {error}
          </div>
        )}

        {loading ? (
          <div className="text-center py-8">Loading...</div>
        ) : (
          <>
            <Card>
              <div className="p-4">
                <h3 className="text-lg font-semibold mb-4">System Prompt</h3>
                <div className="space-y-4">
                  <div>
                    <textarea
                      className="w-full p-2 border rounded dark:bg-gray-700 dark:border-gray-600"
                      value={systemPrompt}
                      onChange={(e) => setSystemPrompt(e.target.value)}
                      rows={3}
                    />
                  </div>
                  <div>
                    <Button onClick={updateSystemPrompt}>Update System Prompt</Button>
                  </div>
                </div>
              </div>
            </Card>

            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mt-6 mb-4">
              LLM Services
            </h2>

            <div className="grid gap-4">
              {providerConfigs.length === 0 ? (
                <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                  No LLM providers found.
                </div>
              ) : (
                providerConfigs.map((providerConfig) => {
                  const provider = providers[providerConfig.id];
                  if (!provider) return null;

                  return (
                    <Card key={providerConfig.id}>
                      <div className="p-4">
                        <div className="flex items-center justify-between mb-4">
                          <div className="flex items-center space-x-4">
                            <div className="relative inline-block w-12 h-6 mr-2">
                              <input
                                type="checkbox"
                                id={`toggle-${providerConfig.id}`}
                                className="sr-only"
                                checked={providerConfig.enabled}
                                onChange={() => toggleProvider(providerConfig.id)}
                              />
                              <label
                                htmlFor={`toggle-${providerConfig.id}`}
                                className={`absolute inset-0 rounded-full cursor-pointer transition-colors duration-300 ${
                                  providerConfig.enabled
                                    ? 'bg-green-500'
                                    : 'bg-gray-300 dark:bg-gray-600'
                                }`}
                              >
                                <span
                                  className={`absolute inset-y-0 left-0 w-6 h-6 bg-white rounded-full shadow transform transition-transform duration-300 ${
                                    providerConfig.enabled ? 'translate-x-6' : 'translate-x-0'
                                  }`}
                                />
                              </label>
                            </div>
                            <div>
                              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                                {providerConfig.id.charAt(0).toUpperCase() +
                                  providerConfig.id.slice(1)}
                              </h3>
                              <p className="text-sm text-gray-500 dark:text-gray-400">
                                {providerConfig.enabled ? (
                                  <span className="text-green-600 dark:text-green-400">Active</span>
                                ) : (
                                  <span>Inactive</span>
                                )}
                              </p>
                            </div>
                          </div>
                        </div>

                        <div className="space-y-4">
                          {provider.requires_api_key && (
                            <div>
                              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                API Key
                              </label>
                              <div className="mt-1 flex">
                                <input
                                  type="password"
                                  className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 dark:bg-gray-700 dark:border-gray-600"
                                  value={providerConfig.api_key}
                                  onChange={(e) => updateApiKey(providerConfig.id, e.target.value)}
                                  placeholder="Enter API key"
                                />
                              </div>
                            </div>
                          )}

                          {provider.requires_base_url && (
                            <div>
                              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                Base URL
                              </label>
                              <div className="mt-1 flex">
                                <input
                                  type="text"
                                  className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 dark:bg-gray-700 dark:border-gray-600"
                                  value={providerConfig.base_url}
                                  onChange={(e) => updateBaseUrl(providerConfig.id, e.target.value)}
                                  placeholder="e.g., http://localhost:11434"
                                />
                              </div>
                            </div>
                          )}

                          {/* Poll Models and Save Buttons */}
                          <div className="flex space-x-2 mt-4">
                            {(provider.requires_api_key || provider.requires_base_url) && (
                              <Button
                                variant="outline"
                                onClick={() => pollAvailableModels(providerConfig.id)}
                                disabled={providerConfig.isPolling}
                              >
                                {providerConfig.isPolling ? 'Polling...' : 'Poll Models'}
                              </Button>
                            )}

                            <Button onClick={() => saveProviderConfig(providerConfig.id)}>
                              Save Configuration
                            </Button>
                          </div>

                          <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                              Chat Model
                            </label>
                            <div className="mt-1">
                              {providerConfig.isPolling ? (
                                <div className="text-sm text-gray-500 dark:text-gray-400">
                                  Polling for available models...
                                </div>
                              ) : providerConfig.available_models.length > 0 ? (
                                <select
                                  className="w-full p-2 border rounded dark:bg-gray-700 dark:border-gray-600"
                                  value={providerConfig.model}
                                  onChange={(e) => updateModel(providerConfig.id, e.target.value)}
                                >
                                  {providerConfig.available_models.map((model) => (
                                    <option key={model} value={model}>
                                      {model}
                                    </option>
                                  ))}
                                </select>
                              ) : (
                                <div className="flex">
                                  <input
                                    type="text"
                                    className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 dark:bg-gray-700 dark:border-gray-600"
                                    value={providerConfig.model}
                                    onChange={(e) => updateModel(providerConfig.id, e.target.value)}
                                    placeholder={`e.g., ${provider.default_model}`}
                                  />
                                </div>
                              )}
                            </div>
                          </div>

                          <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                              Embedding Model
                            </label>
                            <div className="mt-1">
                              {providerConfig.isPolling ? (
                                <div className="text-sm text-gray-500 dark:text-gray-400">
                                  Polling for available models...
                                </div>
                              ) : providerConfig.embedding_models.length > 0 ? (
                                <select
                                  className="w-full p-2 border rounded dark:bg-gray-700 dark:border-gray-600"
                                  value={providerConfig.embedding_model}
                                  onChange={(e) =>
                                    updateEmbeddingModel(providerConfig.id, e.target.value)
                                  }
                                >
                                  {providerConfig.embedding_models.map((model) => (
                                    <option key={model} value={model}>
                                      {model}
                                    </option>
                                  ))}
                                </select>
                              ) : (
                                <div className="flex">
                                  <input
                                    type="text"
                                    className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 dark:bg-gray-700 dark:border-gray-600"
                                    value={providerConfig.embedding_model}
                                    onChange={(e) =>
                                      updateEmbeddingModel(providerConfig.id, e.target.value)
                                    }
                                    placeholder="e.g., text-embedding-ada-002"
                                  />
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    </Card>
                  );
                })
              )}
            </div>
          </>
        )}
      </div>
    </AdminLayout>
  );
};

export default withAdmin(LLMConfiguration, 'LLM Configuration');
