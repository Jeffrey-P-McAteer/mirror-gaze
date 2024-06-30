
#ifdef MAIN_TASKER
#error "Only include main_tasker.hpp ONCE!"
#endif
#define MAIN_TASKER

// This is embedded into main.cpp and we use the globals & functions from main.cpp.
// bad long-term design philosophy, but amazing proof-of-concept organization when we want to focus
// on controlling the model's high-level logic!

int main_tasker(int argc, char** argv) {

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
    "My name is "+username+". Your name is Mirror. Say hello to me and ask what I want to accomplish."
  );

  std::string user_goal_description = prompt_user();
  if ( !( str_ends_in(user_goal_description, '.') || str_ends_in(user_goal_description, '?') || str_ends_in(user_goal_description, '!') ) ) {
    user_goal_description += ".";
  }

  // Now we internally use the LLM to imagine 3 sub-steps to that.
  std::string llm_idea_subgoals = prompt_llm_and_return_value_silent(
    user_goal_description+"\nIdentify three steps to accomplish this."
  );

  std::cout << "[ DEBUG ] llm_idea_subgoals = " << llm_idea_subgoals << std::endl;

  // Interactively imagine more!
  llm_resp = prompt_llm_and_return_value_interactive(
    llm_idea_subgoals+"\nTell me where and how I can accomplish step one."
  );

  llm_resp = prompt_llm_and_return_value_interactive(
    llm_idea_subgoals+"\nTell me where and how I can accomplish step two."
  );

  llm_resp = prompt_llm_and_return_value_interactive(
    llm_idea_subgoals+"\nTell me where and how I can accomplish step three."
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
