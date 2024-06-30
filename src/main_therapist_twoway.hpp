
#ifdef MAIN_THERAPIST_TWOWAY
#error "Only include main_therapist_twoway.hpp ONCE!"
#endif
#define MAIN_THERAPIST_TWOWAY

// This is embedded into main.cpp and we use the globals & functions from main.cpp.
// bad long-term design philosophy, but amazing proof-of-concept organization when we want to focus
// on controlling the model's high-level logic!

int main_therapist_twoway(int argc, char** argv) {

  // We parse our own (more opinionated for task-at-hand) args
  // and construct gemini args for model selection.
  std::vector<char*> args;
  if (argc > 0) {
    args.push_back(argv[0]);
  }
  if (const char* gemma_tokenizer_spm_file = std::getenv("GEMMA_TOKENIZER_SPM_FILE")) {
    args.push_back((char*)"--tokenizer");
    args.push_back((char*)gemma_tokenizer_spm_file);
  }
  if (const char* gemma_model_sbs_file = std::getenv("GEMMA_MODEL_SBS_FILE")) {
    args.push_back((char*)"--weights");
    args.push_back((char*)gemma_model_sbs_file);
    // Infer model type from file name
    auto file_name = std::filesystem::path(gemma_model_sbs_file).filename().string();
    args.push_back((char*)"--model");
    auto model_name = model_from_file_name(file_name);
    args.push_back((char*) model_name );
  }

  // Hard-coded b/c build.py ensures these numbers are available in our modified copy of the llm
  // max_tokens is everything in-memory, both prompt tokens plus max generated tokens
  args.push_back((char*)"--max_tokens");
  args.push_back((char*)"32768");

  // TODO worth tuning / making a per-prompt parameter?
  // This gives us 12k tokens for prompt and 4096 for responses (original ~1024 for responses)
  args.push_back((char*)"--max_generated_tokens");
  args.push_back((char*)"12288");

  // Default is 0, but 1 means the LLM will keep state data between prompts.
  args.push_back((char*)"--multiturn");
  args.push_back((char*)"1");

  // Temperature: Controls randomness, higher values increase diversity.
  args.push_back((char*)"--temperature");
  args.push_back((char*)"2");


  std::thread llm_t(run_llm_thread, args.size(), args.data());
  std::thread term_size_update_t(update_term_size_globals_thread);

  auto username = get_username_from_env();
  std::string llm_resp;

  llm_resp = prompt_llm_and_return_value_interactive(
    "My name is "+username+". Your name is Mirror. Introduce yourself as a therapist interested in learning about my life's struggles."
  );

  std::string user_problem_description = prompt_user();
  llm_resp = prompt_llm_and_return_value_interactive(
    user_problem_description
  );

  // Continue for as long as our llm-agent is asking the user questions.
  while (str_contains(llm_resp, '?')) {
    user_problem_description = prompt_user();
    llm_resp = prompt_llm_and_return_value_interactive(
      user_problem_description
    );
  }

  llm_resp = prompt_llm_and_return_value_interactive(
    "Tell "+username+" the best thing to do. Make sure they are called to take action that fixes their problem."
  );

  llm_resp = prompt_llm_and_return_value_interactive(
    "Energetically say goodbye to "+username+" and wish them success!"
  );

  exit_requested = true;
  llm_input_queue.push(
    "%q" // quit token
  );

  std::cout << "Goodbye!" << std::endl;

  llm_t.join();
  term_size_update_t.join();

  return 0;
}
